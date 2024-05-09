import argparse
import subprocess
from utils import *
from transformers import AutoTokenizer
from pathlib import Path

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def get_args(parser):
    parser.add_argument('input_folder', type=Path, help='The folder containing the input files')

    return parser

# parse arguments
args = get_args(argparse.ArgumentParser()).parse_args()
CLAMP_MODEL_NAME = 'sander-wood/clamp-small-1024'
TEXT_MODEL_NAME = 'distilroberta-base'
TEXT_LENGTH = 128

# load CLaMP model
model = CLaMP.from_pretrained(CLAMP_MODEL_NAME)
music_length = model.config.max_length
model = model.to(device)
model.eval()

# initialize patchilizer, tokenizer, and softmax
patchilizer = MusicPatchilizer()
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
softmax = torch.nn.Softmax(dim=1)

def compute_values(Q_e, K_e, t=1):
    """
    Compute the values for the attention matrix

    Args:
        Q_e (torch.Tensor): Query embeddings
        K_e (torch.Tensor): Key embeddings
        t (float): Temperature for the softmax
    
    Returns:
        values (torch.Tensor): Values for the attention matrix
    """
    # Normalize the feature representations
    Q_e = torch.nn.functional.normalize(Q_e, dim=1)
    K_e = torch.nn.functional.normalize(K_e, dim=1)

    # Scaled pairwise cosine similarities [1, n]
    logits = torch.mm(Q_e, K_e.T) * torch.exp(torch.tensor(t))
    values = softmax(logits)
    return values.squeeze()


def encoding_data(data, modal):
    """
    Encode the data into ids

    Args:
        data (list): List of strings
        modal (str): "music" or "text"
    
    Returns:
        ids_list (list): List of ids
    """
    ids_list = []
    if modal=="music":
        for item in data:
            patches = patchilizer.encode(item, music_length=music_length, add_eos_patch=True)
            ids_list.append(torch.tensor(patches).reshape(-1))
    else:
        for item in data:
            text_encodings = tokenizer(item, 
                                        return_tensors='pt', 
                                        truncation=True, 
                                        max_length=TEXT_LENGTH)
            ids_list.append(text_encodings['input_ids'].squeeze(0))

    return ids_list


def abc_filter(lines):
    """
    Filter out the metadata from the abc file

    Args:
        lines (list): List of lines in the abc file
    
    Returns:
        music (str): Music string
    """
    music = ""
    for line in lines:
        if line[:2] in ['A:', 'B:', 'C:', 'D:', 'F:', 'G', 'H:', 'N:', 'O:', 'R:', 'r:', 'S:', 'T:', 'W:', 'w:', 'X:', 'Z:'] \
        or line=='\n' \
        or (line.startswith('%') and not line.startswith('%%score')):
            continue
        else:
            if "%" in line and not line.startswith('%%score'):
                line = "%".join(line.split('%')[:-1])
                music += line[:-1] + '\n'
            else:
                music += line + '\n'
    return music


def load_music(filename):
    """
    Load the music from the xml file

    Args:
        filename (str): Path to the xml file

    Returns:
        music (str): Music string
    """
    # p = subprocess.Popen('cmd /u /c python inference/xml2abc.py -m 2 -c 6 -x "'+filename+'"', stdout=subprocess.PIPE)
    # result = p.communicate()
    # print(f'python inference/xml2abc.py -m 2 -c 6 -x {filename}')
    if filename.suffix == '.mxl':
        p = subprocess.Popen(f'python inference/xml2abc.py -m 2 -c 6 -x "{filename}"', stdout=subprocess.PIPE, shell=True)
        result = p.communicate()
        output = result[0].decode('utf-8').replace('\r', '')
        music = unidecode(output).split('\n')
        music = abc_filter(music)
    elif filename.suffix == '.mid':
        try:
            subprocess.check_output(f'mscore3 -o "{filename.with_suffix(".mxl")}" "{filename}"', shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise e
        p = subprocess.Popen(f'python inference/xml2abc.py -m 2 -c 6 -x "{filename.with_suffix(".mxl")}"', stdout=subprocess.PIPE, shell=True)
        result = p.communicate()
        output = result[0].decode('utf-8').replace('\r', '')
        music = unidecode(output).split('\n')
        music = abc_filter(music)
        # save music as text
        with open(f'{filename.with_suffix(".txt")}', 'w') as f:
            f.write(music)
        return music



def get_features(ids_list, modal):
    """
    Get the features from the CLaMP model

    Args:
        ids_list (list): List of ids
        modal (str): "music" or "text"
    
    Returns:
        features_list (torch.Tensor): Tensor of features with a shape of (batch_size, hidden_size)
    """
    features_list = []
    print("Extracting "+modal+" features...")
    with torch.no_grad():
        for ids in tqdm(ids_list):
            ids = ids.unsqueeze(0)
            if modal=="text":
                masks = torch.tensor([1]*len(ids[0])).unsqueeze(0)
                features = model.text_enc(ids.to(device), attention_mask=masks.to(device))['last_hidden_state']
                features = model.avg_pooling(features, masks)
                features = model.text_proj(features)
            else:
                masks = torch.tensor([1]*(int(len(ids[0])/PATCH_LENGTH))).unsqueeze(0)
                features = model.music_enc(ids, masks)['last_hidden_state']
                features = model.avg_pooling(features, masks)
                features = model.music_proj(features)

            features_list.append(features[0])
    
    return torch.stack(features_list).to(device)


if __name__ == "__main__":
    target_folder = args.input_folder
    midi_files = list(target_folder.glob('*.mid'))
    for midi_file in midi_files:
        # load query
        try:
            query = load_music(midi_file)
        except:
            continue
        query = unidecode(query)

        # encode query
        query_ids = encoding_data([query], 'music')
        query_feature = get_features(query_ids, 'music')
        torch.save(query_feature, target_folder / f'{midi_file.stem}.pt')
