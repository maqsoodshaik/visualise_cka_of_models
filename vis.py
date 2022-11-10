from transformers import AutoModelForAudioClassification
import torch
from nn_speech_models import SpeechClassifier,FeedforwardClassifier,ConvSpeechEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_checkpoint = "facebook/wav2vec2-base"

label2id, id2label,label2id_int = dict(), dict(),dict()
import matplotlib.pyplot as plt
import yaml
lis_mx = []
names = []
import seaborn as sns
# from CKA import CKA, CudaCKA
# np_cka = CKA()
def print_diff(model,fine_model):
    for i,j in zip(model.named_parameters(),fine_model.named_parameters()):
        # print(f"max size:{max(abs(i).reshape(-1))}  ,abs dif:{max(abs(i-j).reshape(-1))},  shape{i.shape},ratio{max((abs((i-j)/(i+0.00000000000000000009))).reshape(-1))}")
        # lis_mx.append(max((abs((i-j)/(i+0.00000000000000000009))).reshape(-1)).item())
        w1=i[1].detach().numpy()
        w2=j[1].detach().numpy()
        if len(w1.shape) == 1 or w1.shape[0]==1:
            w1 = w1.reshape(-1,1)
            w2 = w2.reshape(-1,1)
        else:
            w1 = w1.reshape(w1.shape[0],-1)
            w2 = w2.reshape(w1.shape[0],-1)
        vL = np_cka.linear_CKA(w1, w2)
        print(f"{vL}")
        lis_mx.append(vL)
        names.append(i[0])
labels =["French","German","Dutch"]
label2id, id2label,label2id_int = dict(), dict(),dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
num_labels = len(id2label)

# print_diff(model.wav2vec2,fine_model.wav2vec2)



import datasets


# datasets.config.DOWNLOADED_DATASETS_PATH = Path('/corpora/fleurs/')


###fleaurs
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
#fleurs
# dataset_name = "fleurs"
# configs = ['fr_fr','de_de','nl_nl']
# list_datasets_validation = []
# for i in configs:   
#     dataset_validation = load_dataset("google/fleurs",i,split = "train")
#     # dataset_validation = Dataset.from_dict(dataset_validation[:int(len(dataset_validation)*(3/4))])
#     list_datasets_validation.append(dataset_validation)
# dataset_validation = concatenate_datasets(
#         list_datasets_validation
#     )
from transformers import AutoFeatureExtractor,AutoModel
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
# def preprocess_function_f(examples):
#     audio_arrays = [x["array"] for x in examples["audio"]]
#     inputs = feature_extractor(
#         audio_arrays, 
#         sampling_rate=feature_extractor.sampling_rate, 
#         max_length=int(feature_extractor.sampling_rate * 5), 
#         truncation=True,
#         padding=True 
#     )
#     inputs["labels"] = [label2id_int[image] for image in examples["language"]]
#     return inputs
# encoded_dataset_validation = dataset_validation.map(preprocess_function_f, remove_columns=["id","num_samples", "path", "audio", "transcription", "raw_transcription", "gender", "lang_id", "language", "lang_group_id"], batched=True)
#multilingua
dataset_name = "multilingual_librispeech"
configs_o = ['french', 'german', 'dutch']
list_datasets_validation_o = []
for val,i in enumerate(configs_o):   
    dataset_validation = load_dataset("facebook/multilingual_librispeech",i,split = "validation")
    dataset_validation = dataset_validation.add_column("labels",[val]*len(dataset_validation))
    list_datasets_validation_o.append(dataset_validation)
dataset_validation_o = concatenate_datasets(
        list_datasets_validation_o
    )
# """We can then write the function that will preprocess our samples. We just feed them to the `feature_extractor` with the argument `truncation=True`, as well as the maximum sample length. This will ensure that very long inputs like the ones in the `_silence_` class can be safely batched."""
max_duration = 10
def preprocess_function_o(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    return inputs
encoded_dataset_validation_multi = dataset_validation_o.map(preprocess_function_o, remove_columns=['file','audio','text','speaker_id','chapter_id','id'], batched=True)

#voxlingua

configs = ['fr','de','nl']
list_datasets_train = []
list_datasets_validation = []
for val,i in enumerate(configs):   
    # write('output_voxlingua.wav', 16000, dataset_train[0]["audio"]["array"])
    dataset_validation = load_dataset("/corpora/voxlingua/",data_dir=i,split = "validation")
    # dataset_validation = dataset_validation.add_column("labels",[val]*len(dataset_validation))

    list_datasets_validation.append(dataset_validation)
dataset_validation = concatenate_datasets(
        list_datasets_validation
    )
labels = configs
label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    label2id_int[label] = i
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True,
        padding=True 
    )
    return inputs
encoded_dataset_validation_vox = dataset_validation.map(preprocess_function, remove_columns=["audio","label"], batched=True)

from torch.utils.data import DataLoader
encoded_dataset_validation_vox.set_format("torch")
encoded_dataset_validation_multi.set_format("torch")
# model.to(device)
torch.manual_seed(0)
eval_dataloader_vox = DataLoader(encoded_dataset_validation_vox, batch_size=4, shuffle=False,drop_last=True)
eval_dataloader_multi = DataLoader(encoded_dataset_validation_multi, batch_size=4, shuffle=False,drop_last=True)


###cnn

nn_speech_encoder_source = AutoModel.from_pretrained(
    "facebook/wav2vec2-base", 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
# def print_param(model):
#     for par in model.parameters():
#         print(par)
nn_speech_encoder_source.feature_projection.projection.out_features=13
# nn_speech_encoder_source.feature_extractor.conv_layers[6].conv.out_channels=13
extractor = nn_speech_encoder_source.feature_extractor
prj = nn_speech_encoder_source.feature_projection 
# obtain yml config file from cmd line and print out content
import sys
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")
config_file_path = sys.argv[1] # e.g., '/speech_cls/config_1.yml'
config_args = yaml.safe_load(open(config_file_path))
# initialize speech encoder
if config_args['encoder_arch']['encoder_model'] == 'ConvEncoder':
    nn_speech_encoder = ConvSpeechEncoder(
        spectral_dim=config_args['encoder_arch']['spectral_dim'],
        num_channels=config_args['encoder_arch']['num_channels'],
        filter_sizes=config_args['encoder_arch']['filter_sizes'],
        stride_steps=config_args['encoder_arch']['stride_steps'],
        pooling_type=config_args['encoder_arch']['pooling_type'],
        dropout_frames=config_args['encoder_arch']['frame_dropout'],
        dropout_spectral_features=config_args['encoder_arch']['feature_dropout'],
        signal_dropout_prob=config_args['encoder_arch']['signal_dropout_prob']
    )

else:
    raise NotImplementedError
# initialize main task classifier ...
nn_task_classifier = FeedforwardClassifier(
    num_classes= config_args['classifier_arch']['num_classes'], # or len(label_set)
    input_dim=config_args['classifier_arch']['input_dim'],
    hidden_dim=config_args['classifier_arch']['hidden_dim'],
    num_layers=config_args['classifier_arch']['num_layers'],
    unit_dropout=config_args['classifier_arch']['unit_dropout'],
    dropout_prob=config_args['classifier_arch']['dropout_prob']
)

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")
# initialize end-2-end LID classifier ...
model_multi_cnn = SpeechClassifier(
    extractor = extractor,
    projector=prj,
    speech_segment_encoder=nn_speech_encoder,
    task_classifier=nn_task_classifier
).to(config_args['device'])
model_vox_cnn = SpeechClassifier(
    extractor = extractor,
    projector=prj,
    speech_segment_encoder=nn_speech_encoder,
    task_classifier=nn_task_classifier
).to(config_args['device'])

# print('\nEnd-to-end LID classifier was initialized ...\n',
#     baseline_LID_classifier)




model_vox_cnn.load_state_dict(torch.load(f'/saved_model/voxlingua_best_model.ckpt'))
model_multi_cnn.load_state_dict(torch.load(f'/saved_model/multilingual_librispeech_best_model.ckpt'))

model_pretrain  = "facebook/wav2vec2-base"
model_vox_wop= "/wop/wav2vec2-basefrdenlvoxlingua_bestmodel"
model_vox_pretrain = "/pretrained/wav2vec2-basefrdenlvoxlingua_bestmodel"
model_multi_pretrain = "/pretrained/wav2vec2-basefrenchgermandutchmultilingual_librispeech_bestmodel"
model_multi_wop = "/wop/wav2vec2-basefrenchgermandutchmultilingual_librispeech_bestmodel"
# model_1 = AutoModelForAudioClassification.from_pretrained(
#     model_multi_wop, 
#      output_hidden_states=True,
#     num_labels=num_labels,
#     label2id=label2id,
#     id2label=id2label,
# )
# model_2 = AutoModelForAudioClassification.from_pretrained(
#    model_vox_wop, 
#     output_hidden_states=True,
#     num_labels=num_labels,
#     label2id=label2id,
#     id2label=id2label,
# )
# breakpoint()
from torch_cka import CKA
lyrs_transformer_conv= ["wav2vec2.feature_extractor.conv_layers.0.conv",
"wav2vec2.feature_extractor.conv_layers.1.conv",
"wav2vec2.feature_extractor.conv_layers.2.conv",
"wav2vec2.feature_extractor.conv_layers.3.conv",
"wav2vec2.feature_extractor.conv_layers.4.conv",
"wav2vec2.feature_extractor.conv_layers.5.conv",]
lyrs_transformer_fw=[
"wav2vec2.encoder.layers.0.final_layer_norm",
"wav2vec2.encoder.layers.1.final_layer_norm",
"wav2vec2.encoder.layers.2.final_layer_norm",
"wav2vec2.encoder.layers.3.final_layer_norm",
"wav2vec2.encoder.layers.4.final_layer_norm",
"wav2vec2.encoder.layers.5.final_layer_norm",
"wav2vec2.encoder.layers.6.final_layer_norm",
"wav2vec2.encoder.layers.7.final_layer_norm",
"wav2vec2.encoder.layers.8.final_layer_norm",
"wav2vec2.encoder.layers.9.final_layer_norm",
"wav2vec2.encoder.layers.10.final_layer_norm",
"wav2vec2.encoder.layers.11.final_layer_norm",]
lyrs_transformer_attention=[
"wav2vec2.encoder.layers.0.attention.out_proj",
"wav2vec2.encoder.layers.1.attention.out_proj",
"wav2vec2.encoder.layers.2.attention.out_proj",
"wav2vec2.encoder.layers.3.attention.out_proj",
"wav2vec2.encoder.layers.4.attention.out_proj",
"wav2vec2.encoder.layers.5.attention.out_proj",
"wav2vec2.encoder.layers.6.attention.out_proj",
"wav2vec2.encoder.layers.7.attention.out_proj",
"wav2vec2.encoder.layers.8.attention.out_proj",
"wav2vec2.encoder.layers.9.attention.out_proj",
"wav2vec2.encoder.layers.10.attention.out_proj",
"wav2vec2.encoder.layers.11.attention.out_proj",]
lyrs_transformer_classifier=[
"projector",
"classifier"
]
lyrs_transformer = lyrs_transformer_conv+lyrs_transformer_attention+lyrs_transformer_fw+lyrs_transformer_classifier

lyrs_cnn_conv = ["extractor.conv_layers.0.conv",
"extractor.conv_layers.1.conv",
"extractor.conv_layers.2.conv",
"extractor.conv_layers.3.conv",
"extractor.conv_layers.4.conv",
"extractor.conv_layers.5.conv",]
lyrs_cnn_attention_fw = [
"speech_encoder.conv1",
"speech_encoder.conv2",
"speech_encoder.conv3",]
lyrs_cnn_classifier = [
"task_classifier._classifier.fc1",
"task_classifier._classifier.fc_last",
"task_classifier._classifier.logits"]
lyrs_cnn = lyrs_cnn_conv+lyrs_cnn_attention_fw+lyrs_cnn_classifier

multilingual_librispeech = "multilingual_librispeech"
vox_lingua_pretrained = "vox_lingua_pretrained"
pretrain = "pretrain"
multi_libre_wop = "multi_libre_wop"
vox_lingua_wop = "vox_lingua_wop"
multi_libre_cnn = "multi_libre_cnn"
vox_lingua_cnn= "vox_lingua_cnn"
cka = CKA(model_multi_cnn,model_vox_cnn,
          model1_name=multi_libre_cnn,   # good idea to provide names to avoid confusion
          model2_name=vox_lingua_cnn, 
          model1_layers=lyrs_cnn,
          model2_layers=lyrs_cnn,
          device='cuda')
# breakpoint()
cka.compare(eval_dataloader_multi,eval_dataloader_vox) # secondary dataloader is optional
results = cka.export()
cka.plot_results(save_path=f"./{multi_libre_cnn}_vs_{vox_lingua_cnn}_compare_all.png")
print("end")
