import os
import re
from Semantic_learning_util.attention_controller.TextExtractor import TextExtractor
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="args for Z-sampling")

    parser.add_argument('--folder_path', type=str, default=f"examples", help='folder path')
    parser.add_argument('--instance', type=str, default="12", help='instance path in folder')
    parser.add_argument('--model_path', type=str, default="/home/skl/hytidel/model/stabilityai/stable-diffusion-2-1-base", help='pretrain model path.')
    parser.add_argument('--seed', type=int, default=None)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    arg = get_args()

    folder_path = f"{arg.folder_path}/{arg.instance}"
    instance_data_path = folder_path + f"/img.jpg"
    instance_mask_path = []

    pattern = re.compile(r"^mask\d+\.png$")

    for filename in os.listdir(folder_path):
        if pattern.match(filename):
            instance_mask_path.append(os.path.join(folder_path, filename))

    extractor = TextExtractor(
                pretrained_model_name_or_path=arg.model_path,
                instance_data_path=instance_data_path,
                instance_mask_path=instance_mask_path,
                object_name="chair",
                class_data_dir="Semantic_learning_util/inputs/data_dir_chairs",
                class_prompt="a photo of a chair",
                seed=arg.seed, # None
                num_of_assets=len(instance_mask_path),
                ## set training steps
                train_steps=600,  # 600
                reward_steps=200,
                DB_training_steps=400,
                initial_learning_rate=5e-3,
                ## init tokens 
                initializer_tokens=[],
                trash_initializer_token = "postrue",
                log_steps=50,
                with_prior_preservation=False,
                apply_masked_loss=True,    
                attn_layers=[7, 8, 9, 10, 11, 12],
                extract_layers = 500,
                lambda_attention = 0,
                # reward para
                reg_weight_reward = 1e-2,
                reward_alpha = 0.5,
                # punish para
                reg_weight_punish = 2e-2,   
                sim_bas = False,
                output_dir = "tmp",    
                use_union_sample = True,
                rec_weight = 1.0,
                trash_weight = 5e-3,
                use_sd_loss=True,
                use_trashbin=False,
                need_DB = False,
                )
    
    extractor.train()

    extractor.DreamBooth(
        out_model_dir="Model_Ours",
        load_path="tmp",
        lora_out_dir=f"{arg.instance}", 
        train_text_embeddings=True,
        use_union_sample=True,
    )
    