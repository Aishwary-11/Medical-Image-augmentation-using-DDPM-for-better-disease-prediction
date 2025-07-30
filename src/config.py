import torch

# General settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_PATH = "C:\\Users\\Aishwary\\PycharmProjects\\PythonProject\\dataset\\chest_xray\\diesease"
TRAIN_DIR = "C:\\Users\\Aishwary\\PycharmProjects\\PythonProject\\dataset\\chest_xray\\chest_xray\\train"
VAL_DIR = "C:\\Users\\Aishwary\\PycharmProjects\\PythonProject\\dataset\\chest_xray\\chest_xray\\val"
TEST_DIR = "C:\\Users\\Aishwary\\PycharmProjects\\PythonProject\\dataset\\chest_xray\\chest_xray\\test"
SYNTHETIC_PATH = "generated_chest_xrays.png"
SAVE_DIR = "./ddpm_chest_xray_output"

# DDPM settings
DDPM_NUM_EPOCHS = 50
DDPM_BATCH_SIZE = 4
DDPM_IMAGE_SIZE = 256
DDPM_LEARNING_RATE = 1e-4
DDPM_MAX_TRAIN_IMAGES = 14000
DDPM_MAX_VAL_IMAGES = 5000
DDPM_BASE_UNET_CHANNELS = 64
DDPM_TIME_EMBEDDING_DIM = 256
DDPM_CHECKPOINT_INTERVAL = 10
DDPM_SAMPLE_INTERVAL = 10

# Classifier settings
CLASSIFIER_MODEL_NAME = 'resnet50'
CLASSIFIER_NUM_EPOCHS = 10
CLASSIFIER_BATCH_SIZE = 8
CLASSIFIER_LEARNING_RATE = 0.001
CLASSIFIER_WEIGHT_DECAY = 1e-4
CLASSIFIER_STEP_SIZE = 7
CLASSIFIER_GAMMA = 0.1
