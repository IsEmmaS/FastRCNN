import torch.utils.data as data
from torch_snippets import *

IMAGE_ROOT = 'images'
DF_RAW = df = pd.read_csv('data_boxes.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pd.set_option('display.width', None)
print('df.csv Structure\n', DF_RAW.head())

label2target = {
    label: target + 1 for target, label in enumerate(DF_RAW['LabelName'].unique())
}
label2target['background'] = 0  # Use dict transport label's name to number target

target2label = {
    target: label for label, target in label2target.items()
}
background_class = label2target['background']  # Same with label to target.
num_classes = len(label2target)


def preprocess_image(image):
    """
    Preprocess the image, Only 'to_tensor' and 'permute' are added.
    Example:
        torch.Size([2, 3, 5])
        torch.permute(x, (2, 0, 1))
        torch.Size([5, 2, 3])
    """
    image = torch.tensor(image).permute(2, 0, 1)
    return image.to(device, dtype=torch.float)


class OpenDataset(data.Dataset):
    w, h = 224, 224

    def __init__(self, dataframe, img_path=IMAGE_ROOT):
        self.img_path = img_path
        self.files = Glob(self.img_path + '/*')
        self.df = dataframe
        self.image_infos = dataframe.ImageID.unique()

    def __getitem__(self, index):
        """
        get image use path + ImageID
        :return: image ,target
        """

        img_id = self.image_infos[index]
        img_path = find(img_id, self.files)

        img = Image.open(img_path).convert('RGB')
        img = np.array(img.resize(size=(self.w, self.h),
                                  resample=Image.BILINEAR)) / 255.
        data = df[df['ImageID'] == img_id]
        labels = data['LabelName'].values.tolist()

        data = data[['XMin', 'YMin', 'XMax', 'YMax']].values

        # Box value was normalized
        data[:, [0, 2]] *= self.w
        data[:, [1, 3]] *= self.h
        boxes = data.astype(np.uint32).tolist()

        # Torch F-RCNN expects ground truths as a dictionary of tensor.
        target = {
            "boxes": torch.Tensor(boxes).float(),
            "labels": torch.Tensor([label2target[i] for i in labels]).long()
        }

        image = preprocess_image(img)

        return image, target

    def __len__(self):
        return len(self.image_infos)

    def collate_fn(self, batch):
        return tuple(zip(*batch))


train_index, test_index = train_test_split(df.ImageID.unique(), test_size=0.1, random_state=37)
train_dataframe, test_dataframe = (
    df[df['ImageID'].isin(train_index)]
    , df[df['ImageID'].isin(test_index)]
)

train_dataset, test_dataset = OpenDataset(dataframe=train_dataframe), OpenDataset(dataframe=test_dataframe)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    collate_fn=train_dataset.collate_fn,
    drop_last=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=4,
    collate_fn=test_dataset.collate_fn,
    drop_last=True
)
