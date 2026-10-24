"""Training-only standardization and two independent ridge mappings."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class CrossModalRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, images, expressions):
        x, y = np.asarray(images), np.asarray(expressions)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Expected at least two paired rows in two feature matrices")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training features must be finite")
        self.image_scaler = StandardScaler().fit(x)
        self.expression_scaler = StandardScaler().fit(y)
        x = self.image_scaler.transform(x); y = self.expression_scaler.transform(y)
        self.i2e = Ridge(alpha=self.alpha, fit_intercept=True, solver="cholesky").fit(x, y)
        self.e2i = Ridge(alpha=self.alpha, fit_intercept=True, solver="cholesky").fit(y, x)
        return self

    def predict(self, features, direction):
        if direction == "i2e":
            return self.i2e.predict(self.image_scaler.transform(features))
        if direction == "e2i":
            return self.e2i.predict(self.expression_scaler.transform(features))
        raise ValueError("direction must be i2e or e2i")

    def gallery(self, features, modality):
        if modality == "image":
            return self.image_scaler.transform(features)
        if modality == "expression":
            return self.expression_scaler.transform(features)
        raise ValueError("modality must be image or expression")
