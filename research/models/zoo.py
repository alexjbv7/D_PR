"""
Models Module
=============
Interfaz unificada para múltiples modelos. Permite swap rápido entre modelos
sin tocar el código de backtesting/ejecución.

Patrón Strategy: todos los modelos implementan fit/predict/predict_proba.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)


# ============================================================================
# INTERFAZ ABSTRACTA
# ============================================================================

class BaseModel(ABC):
    """Interfaz común para todos los modelos. Permite intercambiarlos sin tocar
    el resto del pipeline."""

    name: str = "base"

    def __init__(self, **params):
        self.params = params
        self.model = None
        self.scaler = None
        self.feature_names_ = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> 'BaseModel':
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades para clasificación. Override si aplica."""
        raise NotImplementedError

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Probabilidades SIN calibrar (scores directos del modelo).
        Usado para calibration reports. Override en subclases.
        Default: mismo que predict_proba.
        """
        return self.predict_proba(X)

    def calibrate(self, X_calib, y_calib, method: str = "sigmoid"):
        """
        Calibración de probabilidades con set de calibración separado.
        Implementación base reutilizable. Subclases pueden sobrescribir.

        Usa predict_proba_raw() para obtener scores sin calibrar, luego
        ajusta un IsotonicCalibrator via fit_from_proba().
        """
        from models.calibration import IsotonicCalibrator
        if self.model is None:
            raise RuntimeError("Llama a .fit() antes de .calibrate()")

        raw_proba = self.predict_proba_raw(X_calib)
        y_arr = np.asarray(y_calib)

        # Mapear labels originales → 0..K-1
        label_map = getattr(self, 'label_map_', None)
        if label_map is not None:
            y_mapped = pd.Series(y_arr).map(label_map).values
        else:
            unique_labels = sorted(np.unique(y_arr))
            label_map_local = {lab: i for i, lab in enumerate(unique_labels)}
            y_mapped = np.array([label_map_local[v] for v in y_arr])

        self._calibrator = IsotonicCalibrator(method=method)
        self._calibrator.fit_from_proba(raw_proba, y_mapped.astype(int))
        return self

    def feature_importance(self) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str | Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> 'BaseModel':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# 1. BASELINE: LOGISTIC REGRESSION
#    SIEMPRE entrena este primero. Si XGBoost no le gana significativamente,
#    no tienes señal no-lineal real.
# ============================================================================

class LogisticBaseline(BaseModel):
    name = "logistic"

    def __init__(self, C: float = 1.0, class_weight='balanced'):
        super().__init__(C=C, class_weight=class_weight)
        self.scaler = RobustScaler()  # Robust a outliers, importante en finanzas

    def fit(self, X, y, sample_weight=None):
        self.feature_names_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)
        self.model = LogisticRegression(
            C=self.params['C'],
            class_weight=self.params['class_weight'],
            max_iter=2000,
            solver='lbfgs',
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X.values))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X.values))

    def feature_importance(self) -> pd.Series:
        coef = np.abs(self.model.coef_).mean(axis=0)
        return pd.Series(coef, index=self.feature_names_).sort_values(ascending=False)


# ============================================================================
# 2. XGBOOST — EL CABALLO DE BATALLA
# ============================================================================

class XGBoostClassifier(BaseModel):
    """
    XGBoost para clasificación de señales de trading.

    Hiperparámetros conservadores anti-overfitting:
    - max_depth bajo (3-6)
    - subsample y colsample agresivos (0.7-0.8)
    - reg_alpha/reg_lambda > 0
    - early_stopping_rounds en validación
    """
    name = "xgboost"

    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'gamma': 0.1,
        'objective': 'multi:softprob',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }

    def __init__(self, **params):
        merged = {**self.DEFAULT_PARAMS, **params}
        super().__init__(**merged)
        self.label_map_ = None
        self.inv_label_map_ = None
        self._calibrator = None     # IsotonicCalibrator, seteado tras llamar .calibrate()

    def fit(self, X, y, sample_weight=None, eval_set=None, all_classes=None):
        """
        Parameters
        ----------
        all_classes : list opcional. Si se proporciona, fuerza al modelo a aprender
                      con esas clases (incluso si no aparecen todas en y_train).
                      CRITICO en walk-forward para mantener consistencia entre folds.
        """
        self.feature_names_ = list(X.columns)

        # Mapeo de labels (XGBoost requiere clases 0..K-1)
        # FIX: usar `all_classes` si se proporciona, para evitar desalineacion
        # entre folds cuando alguna clase no aparece en el train.
        if all_classes is not None:
            unique_labels = sorted(all_classes)
        else:
            unique_labels = sorted(np.unique(y))

        self.label_map_ = {lab: i for i, lab in enumerate(unique_labels)}
        self.inv_label_map_ = {i: lab for lab, i in self.label_map_.items()}
        y_mapped = pd.Series(y).map(self.label_map_).values

        # Verificacion: si y tiene clases no esperadas, error
        if pd.isna(y_mapped).any():
            unexpected = set(np.unique(y)) - set(unique_labels)
            raise ValueError(f"Clases en y no esperadas: {unexpected}. "
                             f"Esperaba: {unique_labels}")

        params = self.params.copy()
        params['num_class'] = len(unique_labels)

        self.model = xgb.XGBClassifier(**params)

        fit_kwargs = {'sample_weight': sample_weight}
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_mapped = pd.Series(y_val).map(self.label_map_).values
            fit_kwargs['eval_set'] = [(X_val.values, y_val_mapped)]
            fit_kwargs['verbose'] = False

        self.model.fit(X.values, y_mapped, **fit_kwargs)
        return self

    def predict(self, X):
        # Cuando objective='multi:softprob', xgb.predict() devuelve probabilidades
        # (array 2D), no indices de clase. Usamos predict_proba + argmax.
        proba = self.model.predict_proba(X.values)
        preds_idx = np.argmax(proba, axis=1)
        return np.array([self.inv_label_map_[int(p)] for p in preds_idx])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Probabilidades por clase.

        Si el modelo está calibrado (.calibrate() fue llamado), devuelve
        probabilidades calibradas (P(y=c|x) real).

        Si no está calibrado, devuelve softmax de scores de árboles
        (scores relativos, NO probabilidades interpretables directamente).
        """
        X_arr = X[self.feature_names_].values if isinstance(X, pd.DataFrame) else X
        raw_proba = self.model.predict_proba(X_arr)
        if self._calibrator is not None and self._calibrator.is_fitted:
            return self._calibrator.predict_proba_from_raw(raw_proba)
        return raw_proba

    def calibrate(
        self,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        method: str = "isotonic",
    ) -> "XGBoostClassifier":
        """
        Calibra las probabilidades del modelo usando un set de calibración separado.

        DEBE llamarse DESPUÉS de .fit() y ANTES de cualquier uso de predict_proba()
        como probabilidad real.

        El set de calibración debe ser temporalmente POSTERIOR al train y
        ANTERIOR al test. En walk-forward: últimas 20% barras del train.

        Parameters
        ----------
        X_calib : features del set de calibración.
        y_calib : labels ORIGINALES (e.g., {-1, 0, 1}), NO mapeados.
                  El método mapea internamente usando label_map_.
        method  : 'isotonic' (≥80 muestras) | 'sigmoid' (funciona con <80).

        Returns self (permite chaining: model.fit(...).calibrate(...)).
        """
        if self.model is None:
            raise RuntimeError("Llama a .fit() antes de .calibrate()")
        if self.label_map_ is None:
            raise RuntimeError("El modelo no tiene label_map_. ¿Fue entrenado correctamente?")

        from models.calibration import IsotonicCalibrator

        # Mapear labels originales → 0..K-1 (igual que en fit)
        y_mapped = pd.Series(y_calib).map(self.label_map_).values
        if pd.isna(y_mapped).any():
            unexpected = set(np.unique(y_calib)) - set(self.label_map_.keys())
            raise ValueError(
                f"Labels en y_calib no vistos en entrenamiento: {unexpected}"
            )

        X_arr = X_calib[self.feature_names_].values

        self._calibrator = IsotonicCalibrator(method=method)
        self._calibrator.fit(self.model, X_arr, y_mapped)

        logger.info(
            f"XGBoostClassifier calibrado ({method}) "
            f"sobre {len(X_calib)} muestras."
        )
        return self

    @property
    def is_calibrated(self) -> bool:
        """True si predict_proba() devuelve probabilidades calibradas."""
        return self._calibrator is not None and self._calibrator.is_fitted

    def feature_importance(self) -> pd.Series:
        # Gain importance: mejor que weight para evaluar utilidad real
        booster = self.model.get_booster()
        score = booster.get_score(importance_type='gain')
        # Map de f0,f1,... a nombres reales
        importance = {}
        for i, name in enumerate(self.feature_names_):
            importance[name] = score.get(f'f{i}', 0.0)
        return pd.Series(importance).sort_values(ascending=False)


# ============================================================================
# 3. LSTM — Para cuando tienes señal secuencial real
# ============================================================================

class LSTMClassifier(BaseModel):
    """
    LSTM para clasificación de señales.

    USAR SOLO SI:
    - Tienes >100K muestras de entrenamiento
    - XGBoost ya da resultados decentes (LSTM debería mejorar, no descubrir señal)
    - Tienes GPU
    - Has pensado en la arquitectura

    Implementación con PyTorch para tener control total sobre el training loop.
    """
    name = "lstm"

    def __init__(
        self,
        seq_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 50,
        device: str = 'cpu',
    ):
        super().__init__(
            seq_length=seq_length, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            learning_rate=learning_rate, batch_size=batch_size,
            epochs=epochs, device=device,
        )
        self.scaler = RobustScaler()

    def _build_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Convierte (n_samples, n_features) → (n_samples - seq_length, seq_length, n_features)"""
        seq_len = self.params['seq_length']
        X_seq = np.array([X[i:i + seq_len] for i in range(len(X) - seq_len)])
        if y is not None:
            y_seq = y[seq_len:]
            return X_seq, y_seq
        return X_seq

    def fit(self, X, y, sample_weight=None, eval_set=None):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch requerido para LSTM. pip install torch")

        self.feature_names_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)

        # Map labels
        unique_labels = sorted(np.unique(y))
        self.label_map_ = {lab: i for i, lab in enumerate(unique_labels)}
        self.inv_label_map_ = {i: lab for lab, i in self.label_map_.items()}
        y_mapped = np.array([self.label_map_[v] for v in y])

        X_seq, y_seq = self._build_sequences(X_scaled, y_mapped)

        # Modelo — con LayerNorm después del output del LSTM
        # (CS229 DL cheatsheet: BatchNorm estabiliza el entrenamiento;
        #  para secuencias, LayerNorm es el equivalente apropiado porque
        #  normaliza por feature dentro de cada paso temporal, no sobre el batch)
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0
                )
                # LayerNorm sobre la dimensión hidden_size (por paso temporal)
                self.layer_norm = nn.LayerNorm(hidden_size)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                out, _ = self.lstm(x)
                # Normalizar el último paso temporal antes de la capa lineal
                last = self.layer_norm(out[:, -1, :])
                return self.fc(last)

        device = torch.device(self.params['device'])
        self.model = LSTMNet(
            input_size=X.shape[1],
            hidden_size=self.params['hidden_size'],
            num_layers=self.params['num_layers'],
            num_classes=len(unique_labels),
            dropout=self.params['dropout'],
        ).to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['learning_rate'], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.LongTensor(y_seq)
        )
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)

        self.model.train()
        for epoch in range(self.params['epochs']):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([self.inv_label_map_[i] for i in idx])

    def predict_proba(self, X):
        import torch
        X_scaled = self.scaler.transform(X.values)
        X_seq = self._build_sequences(X_scaled)
        device = torch.device(self.params['device'])
        self.model.eval()
        with torch.no_grad():
            xb = torch.FloatTensor(X_seq).to(device)
            logits = self.model(xb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        # Padding al inicio (no podemos predecir las primeras seq_length)
        pad = np.full((self.params['seq_length'], proba.shape[1]), 1.0/proba.shape[1])
        return np.vstack([pad, proba])


# ============================================================================
# 4. DEEP MLP — Cheatsheet CS229 Deep Learning
#    BatchNorm1d + LeakyReLU + Dropout + Adam
#    Usar cuando tengas >10K muestras y XGBoost ya tiene señal.
#    MLP suele ganar a XGBoost cuando los features son todos numéricos
#    bien escalados y el target tiene estructura no-lineal continua.
# ============================================================================

class DeepMLPClassifier(BaseModel):
    """
    MLP profundo para clasificación de señales de trading.

    Arquitectura (CS229 Deep Learning cheatsheet):
      Linear(in, h1) → BatchNorm1d(h1) → LeakyReLU(0.01) → Dropout(p)
      Linear(h1, h2) → BatchNorm1d(h2) → LeakyReLU(0.01) → Dropout(p)
      ...
      Linear(hn, n_classes)

    Normalización con BatchNorm1d ANTES de la activación (pre-activation):
      reduce covariate shift interno, permite learning rates más altos.

    Optimizer: AdamW (Adam + weight decay) — cheatsheet recomienda Adam
    por su adaptatividad, AdamW añade L2 correcta sobre parámetros.

    Anti-overfitting para finanzas:
      - Dropout en cada capa oculta
      - Weight decay (L2 implícito en AdamW)
      - Capas pequeñas [128, 64, 32] por defecto

    USAR SI:
      - XGBoost ya converge con Sharpe > 0.5 OOS
      - Tienes >10K barras de entrenamiento
      - Los features están bien escalados
    """
    name = "deep_mlp"

    DEFAULT_PARAMS = {
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'epochs': 100,
        'patience': 10,           # early stopping patience
        'device': 'cpu',
        'random_state': 42,
    }

    def __init__(self, **params):
        merged = {**self.DEFAULT_PARAMS, **params}
        super().__init__(**merged)
        self.scaler = RobustScaler()
        self.label_map_: dict | None = None
        self.inv_label_map_: dict | None = None
        self._calibrator = None

    # ------------------------------------------------------------------
    # Arquitectura interna
    # ------------------------------------------------------------------

    def _build_net(self, input_dim: int, hidden_dims: list, n_classes: int, dropout: float):
        """Construye el MLP con BatchNorm1d + LeakyReLU + Dropout."""
        import torch.nn as nn

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight=None,
        eval_set=None,
        all_classes=None,
    ) -> "DeepMLPClassifier":
        """
        Parameters
        ----------
        eval_set    : (X_val, y_val) opcional para early stopping.
        all_classes : lista de clases esperadas (mantiene consistencia cross-fold).
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch requerido para DeepMLP. pip install torch")

        torch.manual_seed(self.params['random_state'])
        self.feature_names_ = list(X.columns)

        # Escalar
        X_scaled = self.scaler.fit_transform(X.values).astype(np.float32)

        # Mapeo de labels → 0..K-1
        if all_classes is not None:
            unique_labels = sorted(all_classes)
        else:
            unique_labels = sorted(np.unique(y))

        self.label_map_ = {lab: i for i, lab in enumerate(unique_labels)}
        self.inv_label_map_ = {i: lab for lab, i in self.label_map_.items()}
        y_mapped = pd.Series(y).map(self.label_map_).values.astype(np.int64)

        if pd.isna(y_mapped).any():
            unexpected = set(np.unique(y)) - set(unique_labels)
            raise ValueError(f"Clases en y no esperadas: {unexpected}")

        n_classes = len(unique_labels)
        device = torch.device(self.params['device'])

        # Construir red
        self.model = self._build_net(
            input_dim=X_scaled.shape[1],
            hidden_dims=self.params['hidden_dims'],
            n_classes=n_classes,
            dropout=self.params['dropout'],
        ).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay'],
        )

        # Pérdida con peso de clases (class_weight='balanced' equivalent)
        if sample_weight is not None:
            # Pesos de muestra en CrossEntropyLoss
            class_counts = np.bincount(y_mapped, minlength=n_classes).astype(float)
            class_counts = np.where(class_counts == 0, 1.0, class_counts)
            class_weights = torch.FloatTensor(
                len(y_mapped) / (n_classes * class_counts)
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # Auto-balanceo por frecuencia de clases
            class_counts = np.bincount(y_mapped, minlength=n_classes).astype(float)
            class_counts = np.where(class_counts == 0, 1.0, class_counts)
            class_weights = torch.FloatTensor(
                len(y_mapped) / (n_classes * class_counts)
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Dataset
        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.LongTensor(y_mapped).to(device)
        if sample_weight is not None:
            w_t = torch.FloatTensor(np.asarray(sample_weight, dtype=np.float32)).to(device)
            dataset = TensorDataset(X_t, y_t, w_t)
        else:
            dataset = TensorDataset(X_t, y_t)

        loader = DataLoader(
            dataset,
            batch_size=self.params['batch_size'],
            shuffle=True,
        )

        # Validation para early stopping
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_scaled = self.scaler.transform(X_val.values).astype(np.float32)
            y_val_mapped = pd.Series(y_val).map(self.label_map_).values.astype(np.int64)
            X_vt = torch.FloatTensor(X_val_scaled).to(device)
            y_vt = torch.LongTensor(y_val_mapped).to(device)
            val_loader = DataLoader(TensorDataset(X_vt, y_vt),
                                    batch_size=self.params['batch_size'])

        # Training loop con early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        patience = self.params['patience']

        for epoch in range(self.params['epochs']):
            self.model.train()
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                if len(batch) == 3:
                    xb, yb, wb = batch
                    logits = self.model(xb)
                    # Weighted cross-entropy manual
                    loss = (nn.functional.cross_entropy(logits, yb, reduction='none') * wb).mean()
                else:
                    xb, yb = batch
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                loss.backward()
                # Gradient clipping (cheatsheet: clip gradients para estabilidad)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            # Early stopping sobre validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xv, yv in val_loader:
                        val_loss += criterion(self.model(xv), yv).item()
                val_loss /= len(val_loader)
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping en epoch {epoch}")
                        break

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

        # Restaurar mejor estado
        if best_state is not None:
            self.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        return self

    # ------------------------------------------------------------------
    # predict / predict_proba
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array([self.inv_label_map_[int(i)] for i in idx])

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Softmax del MLP SIN calibración."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch requerido. pip install torch")

        X_arr = X[self.feature_names_].values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_arr).astype(np.float32)
        device = torch.device(self.params['device'])
        self.model.eval()
        with torch.no_grad():
            xb = torch.FloatTensor(X_scaled).to(device)
            logits = self.model(xb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Probabilidades por clase.
        Si calibrado → devuelve probabilidades calibradas.
        Si no → devuelve softmax directo.
        """
        raw = self.predict_proba_raw(X)
        if self._calibrator is not None and self._calibrator.is_fitted:
            return self._calibrator.predict_proba_from_raw(raw)
        return raw

    @property
    def is_calibrated(self) -> bool:
        return self._calibrator is not None and self._calibrator.is_fitted

    # ------------------------------------------------------------------
    # feature_importance — gradient-based saliency (media |∇x| sobre test)
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.Series:
        """
        Importancia por saliency map: |gradiente de la pérdida respecto a x|.
        Proxy de qué features mueven más la predicción.
        Solo disponible si PyTorch está instalado.
        """
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("Modelo no entrenado")
        # Sin datos de referencia, devolvemos pesos de la primera capa como proxy
        try:
            import torch
            first_layer = list(self.model.parameters())[0]  # (h1, input_dim)
            importance = first_layer.abs().mean(dim=0).detach().cpu().numpy()
            return pd.Series(importance, index=self.feature_names_).sort_values(ascending=False)
        except Exception:
            return pd.Series(np.zeros(len(self.feature_names_)), index=self.feature_names_)


# ============================================================================
# FACTORY
# ============================================================================

def get_model(name: str, **kwargs) -> BaseModel:
    registry = {
        'logistic':  LogisticBaseline,
        'xgboost':   XGBoostClassifier,
        'lstm':      LSTMClassifier,
        'deep_mlp':  DeepMLPClassifier,
    }
    if name not in registry:
        raise ValueError(f"Modelo '{name}' no registrado. Disponibles: {list(registry.keys())}")
    return registry[name](**kwargs)
