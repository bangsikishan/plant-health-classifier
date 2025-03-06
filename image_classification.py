import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModelForImageClassification

# Configuration
MODEL_SAVE_PATH = "D:\Machine Learning\image_classification\image_classification_model\distilled_model_1.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
TEMPERATURE = 3.0


def load_models():
    """Load teacher and student models"""
    # Teacher model from HuggingFace
    teacher = AutoModelForImageClassification.from_pretrained(
        "merve/vit-mobilenet-beans-224"
    )
    teacher.eval()

    # Simple student model (MobileNetV2)
    student = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
    student.classifier[1] = nn.Linear(
        student.classifier[1].in_features, 3
    )  # 3 classes for beans dataset

    return teacher, student


def prepare_datasets():
    """Load and prepare beans dataset"""
    dataset = load_dataset("beans")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def process(examples):
        examples["pixel_values"] = [
            transform(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    dataset = dataset.map(process, batched=True)
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    return dataset["train"], dataset["validation"]


def train_model(teacher, student, train_loader, val_loader):
    """Training loop with knowledge distillation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    teacher = teacher.to(device)

    optimizer = optim.Adam(student.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Training
        student.train()
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1} [Training]", leave=False
        )

        for batch in progress_bar:
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward passes
            with torch.no_grad():
                teacher_logits = teacher(inputs).logits

            student_logits = student(inputs)

            # Calculate losses
            loss_ce = ce_loss(student_logits, labels)
            loss_kl = kl_loss(
                torch.log_softmax(student_logits / TEMPERATURE, dim=1),
                torch.softmax(teacher_logits / TEMPERATURE, dim=1),
            )
            total_loss = 0.7 * loss_kl + 0.3 * loss_ce

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=total_loss.item())

        scheduler.step()

        # Validation
        student.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1} [Validation]", leave=False
            ):
                inputs = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = student(inputs)
                loss = ce_loss(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), MODEL_SAVE_PATH)

    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")


def get_prediction(image_path):
    # Load the student model (MobileNetV2)
    student = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)
    student.classifier[1] = nn.Linear(
        student.classifier[1].in_features, 3
    )  # 3 classes for beans dataset

    # Load the trained model weights
    student.load_state_dict(torch.load(MODEL_SAVE_PATH))
    student.eval()  # Set to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Transform the image
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = student(input_batch)

    # Get predicted class
    _, predicted = torch.max(output.data, 1)

    # Define class names (replace with your actual class names)
    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]

    predicted_class = class_names[predicted.item()]

    return predicted_class


def main():
    # teacher, student = load_models()
    # train_set, val_set = prepare_datasets()

    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # train_model(teacher, student, train_loader, val_loader)
    # print(f"Model saved to {MODEL_SAVE_PATH}")

    print(
        get_prediction(
            r"D:\Machine Learning\image_classification\test_images\angular_leaf_spot_test.3.jpg"
        )
    )


if __name__ == "__main__":
    main()
