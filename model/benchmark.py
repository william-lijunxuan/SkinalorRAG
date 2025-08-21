from utils import (
    MMSkinQA,
    SkinCAP,
    Skingpt)

def prepare_benchmark(idx,model,eval_dataset,eval_dataset_path,eval_output_path):
    supported_dataset = ["MMSkinQA","SkinCAP","SKINgpt"]
    if eval_dataset == "MMSkinQA":
        dataset = MMSkinQA(idx,model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "SkinCAP":
        dataset = SkinCAP(idx,model,eval_dataset_path,eval_output_path)
    elif eval_dataset == "SKINgpt":
        dataset = Skingpt(idx,model, eval_dataset_path, eval_output_path)
    else:
        print(f"unknown eval dataset {eval_dataset}, we only support {supported_dataset}")
        dataset = None
    return dataset