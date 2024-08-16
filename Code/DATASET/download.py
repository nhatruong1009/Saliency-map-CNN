import argparse
import subprocess

dictation = {
    "imagenet": 
        {
            "2012": {
                "train": "wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate",
                "valid": "wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate"
                }
        },
    "mimiciv": {
        "2.2": "wget -r -N -c -np --user lengocduc195 --ask-password https://physionet.org/files/mimiciv/2.2/"
    },
    "mimiciii": {
        "1-4": "wget -r -N -c -np --user lengocduc195 --ask-password https://physionet.org/files/mimiciii/1.4/"
    },
    "mimic-cxr-jpg": {
        "2-1-0": "wget -r -N -c -np --user lengocduc195 --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
    },
    "cxr-pro": {
        "1-0-0": "wget -r -N -c -np --user lengocduc195 --ask-password https://physionet.org/files/cxr-pro/1.0.0/"
    }
}

def main():
    parser = argparse.ArgumentParser(description='Download data using wget based on input')
    parser.add_argument('-dataset', type=str, help='Input to determine what to download')
    parser.add_argument('-version', type=str, help='Input to determine what to download')
    parser.add_argument('-split', type=str, help='Input to determine what to download', default=None)
    args = parser.parse_args()

    dataset_tokens = args.dataset
    version_tokens = args.version
    split_tokens = args.split
    if dataset_tokens in dictation.keys():
        if version_tokens in dictation[dataset_tokens].keys():
            if split_tokens==None:
                command = dictation[dataset_tokens][version_tokens]
                print(f'Downloading IMAGENET TRAIN data...')
                subprocess.run(command, shell=True)
                print('Download completed.')
            else:
                command = dictation[dataset_tokens][version_tokens][split_tokens]
                print(f'Downloading IMAGENET TRAIN data...')
                subprocess.run(command, shell=True)
                print('Download completed.')
        
    elif args.input.lower() == '--help':
        print(dictation)
    else:
        print('Input not recognized.')

if __name__ == '__main__':
    main()
