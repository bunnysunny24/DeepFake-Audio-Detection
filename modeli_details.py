import os
import time
import torch
import torch.nn as nn
import torch.cuda
from torchinfo import summary

# Load your model (Replace this with your actual model)
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 112 * 112, 10)  # Assuming input 224x224

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Load or define your model
model = DummyModel()
model_path = "model.pth"  # Save and load model if needed
torch.save(model.state_dict(), model_path)  # Save dummy model

# 1️⃣ Model Size (in MB)
model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"Model Size: {model_size:.2f} MB")

# 2️⃣ Number of Parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

# 3️⃣ FLOPs (Floating Point Operations)
summary(model, input_size=(1, 3, 224, 224))  # Adjust input size as needed

# 4️⃣ Inference Time Measurement
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape if needed
model.eval()
start_time = time.time()
with torch.no_grad():
    _ = model(dummy_input)
end_time = time.time()
inference_time = (end_time - start_time) * 1000  # Convert to ms
print(f"Inference Time: {inference_time:.2f} ms")

# 5️⃣ Memory Usage (GPU)
if torch.cuda.is_available():
    model.cuda()
    dummy_input = dummy_input.cuda()
    torch.cuda.empty_cache()
    before_memory = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        _ = model(dummy_input)

    after_memory = torch.cuda.memory_allocated()
    memory_used = (after_memory - before_memory) / (1024 * 1024)
    print(f"Memory Used on GPU: {memory_used:.2f} MB")
else:
    print("CUDA not available, skipping GPU memory test.")
