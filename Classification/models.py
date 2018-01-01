class PredictiveModel:
    def __init__(self, classifier, label_encoder):
        self.classifier = classifier
        self.label_encoder = label_encoder