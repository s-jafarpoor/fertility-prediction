import torch
from torch import nn

class TransformerClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim=2):
        super(TransformerClassification, self).__init__()

        # **لایه اولیه برای تبدیل ویژگی‌ها به ابعاد مخفی مناسب**
        self.fc_input = nn.Linear(input_dim, hidden_dim)

        # **لایه‌های ترنسفورمر**
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # **لایه نهایی خروجی برای دسته‌بندی**
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # **تبدیل ورودی‌ها به فضای مخفی**
        x = self.fc_input(x)  # تبدیل ورودی به hidden_dim
        x = x.unsqueeze(1)  # افزودن بعد sequence_length = 1

        # **پردازش با ترنسفورمر**
        x = self.transformer_encoder(x)

        # **میانگین‌گیری از خروجی‌ها (چون sequence_length=1 است)**
        x = x.mean(dim=1)

        # **لایه خروجی برای دسته‌بندی**
        x = self.fc_output(x)

        return x
