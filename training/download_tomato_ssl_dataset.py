import fiftyone as fo
import fiftyone.zoo as foz



list_classes = """
Tomato
Plant
Houseplant
Flower
Flowerpot
Tree
Vegetable
Fruit
Bell pepper
Cucumber
""".strip().split("\n")


#dataset = foz.load_zoo_dataset(
#    "open-images-v7",
#    split="train",
#    label_types=["detections"],
#    classes=list_classes,
#    max_samples=300000,
#)

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=list_classes,
    max_samples=5000,
)
