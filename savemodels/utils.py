import os
import torch



def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    # WEIGHTS_NAME = "pytorch_model.bin"


    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    # output_model_file = os.path.join(output_dir, WEIGHTS_NAME)

    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    save_dict = model_to_save.state_dict()


    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]

    # torch.save(save_dict, output_model_file)
    model_to_save.save_pretrained(output_dir,
                                  state_dict=save_dict,
                                  safe_serialization=True)

    model_to_save.config.to_json_file(output_config_file)

    tokenizer.save_pretrained(output_dir)

    model_to_save.generation_config.save_pretrained(args.output_dir)



