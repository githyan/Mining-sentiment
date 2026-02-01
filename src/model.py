import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class BERTCNNSentiment_Model(nn.Module):
    def __init__(self, num_classes=3, filter_sizes=(3,4,5), num_filters=128, dropout=0.2) -> None:
        super(BERTCNNSentiment_Model, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=768, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes) + 768, num_classes)

    def forward(self, sequences_output, cls_output):
        X = sequences_output.permute(0, 2, 1)

        convs_out = [F.relu(convs(X)) for convs in self.convs]
        pooled_out = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs_out]

        cnn_features = torch.cat(pooled_out, dim=1)
        combined_features = torch.cat([cnn_features, cls_output], dim=1)

        return self.fc(self.dropout(combined_features))


def initialize_model(num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("indolem/indobert-base-uncased", do_lower_case=True)
    bert = BertModel.from_pretrained("indolem/indobert-base-uncased").to(device)
    classification_model = BERTCNNSentiment_Model(num_classes=num_classes).to(device)

    # for params1, params2 in zip(bert.parameters(), classification_model.parameters()):
    #     params1.requires_grad = True
    #     params2.requires_grad =True

    optimizer = torch.optim.AdamW([
        {"params": bert.parameters(), "lr": 2e-5, "weight_decay": 1e-2},
        {"params": classification_model.parameters(), "lr": 1e-4, "weight_decay": 1e-3}
    ])

    return bert, classification_model, optimizer, device, tokenizer

