use anyhow::Result;
use candle::{Device, IndexOp, Tensor};

use dora_node_api::{dora_core::config::DataId, into_vec, DoraNode, IntoArrow, MetadataParameters};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub mimi_name: String,
    pub moshi_name: String,
    pub tokenizer_name: String,
    pub model: moshi::lm::Config,
}

pub struct Args {
    pub lm_config: moshi::lm::Config,
    pub lm_model_file: std::path::PathBuf,
    pub mimi_model_file: std::path::PathBuf,
    pub text_tokenizer: std::path::PathBuf,
    pub seed: u64,
    pub cfg_alpha: Option<f64>,
}

fn text(
    text_tokenizer: &sentencepiece::SentencePieceProcessor,
    prev_text_token: u32,
    text_token: u32,
    text_start_token: u32,
) -> Option<String> {
    if prev_text_token == text_start_token {
        text_tokenizer.decode_piece_ids(&[text_token]).ok()
    } else {
        let prev_ids = text_tokenizer.decode_piece_ids(&[prev_text_token]).ok();
        let ids = text_tokenizer.decode_piece_ids(&[prev_text_token, text_token]).ok();
        prev_ids.and_then(|prev_ids| {
            ids.map(|ids| {
                if ids.len() > prev_ids.len() {
                    ids[prev_ids.len()..].to_string()
                } else {
                    String::new()
                }
            })
        })
    }
}

pub fn run(args: &Args, dev: &Device) -> Result<()> {
    let dtype = dev.bf16_default_to_f32();
    let lm_config = &args.lm_config;
    tracing::info!(?dtype, ?dev);

    tracing::info!("loading the audio input");

    tracing::info!("loading the lm");
    let lm_model = moshi::lm::load_lm_model(lm_config.clone(), &args.lm_model_file, dtype, dev)?;
    tracing::info!("loading the audio tokenizer");
    let mut mimi = moshi::mimi::load(
        args.mimi_model_file.to_str().unwrap(),
        Some(lm_model.generated_audio_codebooks()),
        dev,
    )?;
    tracing::info!("loading the text tokenizer");
    let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&args.text_tokenizer)?;
    tracing::info!("done loading models");

    let audio_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
        args.seed,
        candle_transformers::generation::Sampling::TopK { k: 250, temperature: 0.8 },
    );
    let text_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
        args.seed,
        candle_transformers::generation::Sampling::TopK { k: 25, temperature: 0.8 },
    );
    let generated_audio_codebooks = lm_config.depformer.as_ref().map_or(8, |v| v.num_slices);

    let conditions = match lm_model.condition_provider() {
        None => None,
        Some(cp) => {
            let conditions = if args.cfg_alpha.is_some() {
                use moshi::conditioner::Condition::AddToInput;
                let AddToInput(c1) = cp.condition_lut("description", "very_good")?;
                let AddToInput(c2) = cp.condition_lut("description", "very_bad")?;
                AddToInput(Tensor::cat(&[c1, c2], 0)?)
            } else {
                cp.condition_lut("description", "very_good")?
            };
            tracing::info!(?conditions, "generated conditions");
            Some(conditions)
        }
    };
    let max_steps = 2500;
    let cfg_alpha = if args.cfg_alpha == Some(1.) { None } else { args.cfg_alpha };
    let mut state = {
        let config = moshi::lm_generate_multistream::Config {
            acoustic_delay: 2,
            audio_vocab_size: lm_config.audio_vocab_size,
            generated_audio_codebooks,
            input_audio_codebooks: lm_config.audio_codebooks - generated_audio_codebooks,
            text_start_token: lm_config.text_out_vocab_size as u32,
            text_eop_token: 0,
            text_pad_token: 3,
        };
        moshi::lm_generate_multistream::State::new(
            lm_model, 200_000, audio_lp, text_lp, None, None, cfg_alpha, config,
        )
    };

    let text_start_token = state.config().text_start_token;
    let mut prev_text_token = text_start_token;
    let mut out_pcms = vec![];
    let mut text_tokens = vec![];
    tracing::info!("starting the inference loop");
    let (mut node, mut event) = DoraNode::init_from_env().unwrap();
    while let Some(event) = event.recv() {
        let data = match event {
            dora_node_api::Event::Input { id, metadata, data } => data,
            dora_node_api::Event::Stop(_) => {
                break;
            }
            _ => {
                tracing::warn!("unexpected event: {event:?}");
                continue;
            }
        };
        let data: Vec<f32> = into_vec(&data).unwrap();
        // println!("received input data of length {}", data.len());
        let len = data.len();
        let in_pcm = Tensor::from_vec(data, (1, 1, len), dev)?;
        let codes = mimi.encode_step(&in_pcm.into())?;
        if let Some(codes) = codes.as_option() {
            let (_b, _codebooks, steps) = codes.dims3()?;
            for step in 0..steps {
                let codes = codes.i((.., .., step..step + 1))?;
                let codes = codes.i((0, .., 0))?.to_vec1::<u32>()?;
                let text_token =
                    state.step_(Some(prev_text_token), &codes, None, None, conditions.as_ref())?;
                if text_token != 0 && text_token != 3 {
                    text_tokens.push(text_token);
                    if let Some(text) =
                        text(&text_tokenizer, prev_text_token, text_token, text_start_token)
                    {
                        use std::io::Write;
                        println!("{text}");
                        std::io::stdout().flush().unwrap();
                    }
                }
                prev_text_token = text_token;
                if let Some(audio_tokens) = state.last_audio_tokens() {
                    let audio_tokens =
                        Tensor::new(&audio_tokens[..generated_audio_codebooks], dev)?
                            .reshape((1, 1, ()))?
                            .t()?;
                    let out_pcm = mimi.decode_step(&audio_tokens.into())?;
                    if let Some(out_pcm) = out_pcm.as_option() {
                        out_pcms.push(out_pcm.clone());
                        let out_pcm = out_pcm.i((0, 0))?.to_vec1::<f32>()?;
                        let mut parameters = MetadataParameters::new();
                        parameters.insert(
                            "sample_rate".to_string(),
                            dora_node_api::Parameter::Integer(24000),
                        );
                        node.send_output(
                            DataId::from("audio".to_string()),
                            parameters,
                            out_pcm.into_arrow(),
                        )
                        .unwrap()
                    }
                }
            }
        }
    }

    let str = text_tokenizer.decode_piece_ids(&text_tokens)?;
    tracing::info!(str, "generated text");
    Ok(())
}
