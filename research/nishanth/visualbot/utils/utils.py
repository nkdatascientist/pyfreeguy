import subprocess, os

def download_coco_dataset(script_path:str):
    process = subprocess.Popen(["bash", script_path], stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode())

def dataset_verification(args):
    if os.path.exists(args.dataset["dataset_path"]):
        return True
    return False

class Dataset_Preparation:
    def __init__(self, args):
        self.args = args
        self.dataset_preparation()

    def dataset_preparation(self):
        if self.args.dataset["dataset_name"] == "COCO2014":
            script_path = self.args.dataset["dataset_path"]
            download_coco_dataset(script_path)

    def __call__(self, txt_model, device):
        assert dataset_verification(self.args), "Dataset not exists"
        return

