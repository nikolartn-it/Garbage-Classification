import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Konvoluciona neuronska mreza za klasifikaciju otpada.
    
    Args:
        num_classes (int): Broj izlaznih klasa (6 za ovaj dataset).
        filters (list): Broj filtera u svakom konvolucionom sloju.
                        Default: [32, 64, 128, 256]
        dropout (float): Dropout verovatnoca pre finalnog Linear sloja.
                         Default: 0.5
    """
    
    def __init__(self, num_classes, filters=None, dropout=0.5):
        super(Model, self).__init__()
        
        if filters is None:
            filters = [32, 64, 128, 256]
        
        self.filters = filters
        self.dropout_rate = dropout
        
        # Konvolucioni slojevi (feature extractor)
        self.features = nn.Sequential(
            # Blok 1: 3 -> filters[0]
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Blok 2: filters[0] -> filters[1]
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Blok 3: filters[1] -> filters[2]
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Blok 4: filters[2] -> filters[3]
            nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Klasifikator
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filters[3], num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(num_classes, config=None):
    """
    Factory funkcija za kreiranje modela iz konfiguracije.
    
    Args:
        num_classes (int): Broj klasa.
        config (dict): Konfiguracija eksperimenta (moze sadrzati 'filters' i 'dropout').
    
    Returns:
        Model: Instanca modela.
    """
    if config is None:
        config = {}
    
    filters = config.get('filters', None)
    dropout = config.get('dropout', 0.5)
    
    model = Model(
        num_classes=num_classes,
        filters=filters,
        dropout=dropout,
    )
    
    return model