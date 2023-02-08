from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import os, json

class GetSummaryWriter:
    """
    "summary":{
        "name": "",
        "log_dir":""
    }
    """
    def __init__(self, args):
        file_name = args.summary["name"]
        dir_name = args.summary["log_dir"]

        self.data, self.tf_files = {}, []
        os.makedirs(dir_name, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(dir_name, file_name))
        
    def insert(self, name, key, value):
        self.writer.add_scalar(name, key, value)

    def export(self):
        assert isinstance(self.writer, SummaryWriter)
        for root, dirs, files in os.walk(self.writer.log_dir):
            for file in files:
                self.tf_files.append(os.path.join(root,file))

        for file_id, file in enumerate(self.tf_files):
            # determine path to folder in which file lies
            path = os.path.join('/'.join(file.split('/')[:-1])) 
            # seperate file created by add_scalar from add_scalars
            name = os.path.join(file.split('/')[-2]) if file_id > 0 else os.path.join('data') 

            event_acc = event_accumulator.EventAccumulator(self.writer.log_dir)
            event_acc.Reload()

            hparam_file = False # I save hparam files as 'hparam/xyz_metric'
            for tag in sorted(event_acc.Tags()["scalars"]):
                if tag.split('/')[0] == 'hparam': 
                    hparam_file=True # check if its a hparam file
                
                step, value = [], []
                for scalar_event in event_acc.Scalars(tag):
                    step.append(scalar_event.step)
                    value.append(scalar_event.value)
                self.data[tag] = (step, value)

            filename = f'{path}/{name}.json'
            try:
                if os.path.exists(filename) and os.path.getsize(filename) != 0 :
                    with open(filename, "x") as file:
                        self.data = json.load(file)
            except FileExistsError:
                pass
                # print(f"File {filename} already exists.")

            # if its not a hparam file and there is something in the self.data -> dump it
            if not hparam_file and bool(self.data): 
                with open(f'{filename}', "w") as f:
                    json.dump(self.data, f)