from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
from pydantic import BaseModel, Field, model_validator, computed_field
import torch
from datasets.arrow_dataset import Dataset
import eliot
from enum import Enum, auto

from biomodels_mcp.config import (
    MODEL_PATH, TOKENIZER_PATH, DEFAULT_DATA_DIR, 
    LOGS_DIR,
    # Direct URL constants instead of FILE_SPECS
    DEFAULT_GENE_ANNOTATION_URL,
    DEFAULT_REFERENCE_GENOME_URL,
    DEFAULT_VCF_URL
)
from biomodels_mcp.downloads import (
    download_file
)
from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter

# Define an enum for dataset subsets
class DatasetSubset(str, Enum):
    EXPRESSION = "expression"
    CODING_PATHOGENICITY = "coding_pathogenicity"
    NON_CODING_PATHOGENICITY = "non_coding_pathogenicity"
    COMMON_VS_RARE = "common_vs_rare"
    MEQTL = "meqtl"
    SQTL = "sqtl"

class BioFM(BaseModel):
    """Pydantic model representing BioFM model operations"""
    # Basic configuration
    data_dir: Path = Field(default_factory=lambda: DEFAULT_DATA_DIR, 
                          description="Directory to store/find data files")
    logs_dir: Path = Field(default_factory=lambda: LOGS_DIR, 
                         description="Directory to store log files")
    max_variants: int = Field(default=200, 
                            description="Maximum number of variants to process from VCF file")
    force_download: bool = Field(default=False, 
                               description="Force download of data files even if they already exist")
    
    # Model configuration
    model_path: str = Field(default=MODEL_PATH, 
                          description="Path to the pre-trained BioFM model")
    tokenizer_path: str = Field(default=TOKENIZER_PATH, 
                              description="Path to the tokenizer")
    
    # Data file paths - can be local paths or remote URLs
    gene_annotation_url: str = Field(
        default=DEFAULT_GENE_ANNOTATION_URL,
        description="URL or local path to gene annotation file"
    )
    
    reference_genome_url: str = Field(
        default=DEFAULT_REFERENCE_GENOME_URL,
        description="URL or local path to reference genome file"
    )
    
    vcf_url: Optional[str] = Field(
        default=DEFAULT_VCF_URL,
        description="URL or local path to VCF file"
    )
    
    # Local paths (set automatically)
    gene_annotation_path: Optional[Path] = Field(default=None, exclude=True)
    reference_genome_path: Optional[Path] = Field(default=None, exclude=True)
    vcf_path: Optional[Path] = Field(default=None, exclude=True)
    
    # Results
    embeddings: Optional[Dict[str, np.ndarray]] = Field(default=None, exclude=True)
    dataset: Optional[Dataset] = Field(default=None, exclude=True)
    
    # Private fields (without leading underscores to work with Pydantic)
    model_instance: Optional[AnnotatedModel] = Field(default=None, exclude=True)
    tokenizer_instance: Optional[AnnotationTokenizer] = Field(default=None, exclude=True)
    embedder_instance: Optional[Embedder] = Field(default=None, exclude=True)
    vcf_converter_instance: Optional[VCFConverter] = Field(default=None, exclude=True)
    
    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode='after')
    def ensure_directories_exist(self) -> 'BioFM':
        """Ensure all necessary directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self
    
    @model_validator(mode='after')
    def prepare_file_paths(self) -> 'BioFM':
        """Set up file paths based on URLs/paths provided"""
        with eliot.start_action(action_type="prepare_file_paths") as action:
            # Gene annotation file
            if self.gene_annotation_path is None:
                self._download_or_set_path(
                    url=self.gene_annotation_url, 
                    path_attr="gene_annotation_path",
                    file_type="gene_annotation"
                )
                action.log(message_type="gene_annotation_path_set", 
                         path=str(self.gene_annotation_path) if self.gene_annotation_path else None)
            
            # Reference genome file
            if self.reference_genome_path is None:
                self._download_or_set_path(
                    url=self.reference_genome_url, 
                    path_attr="reference_genome_path",
                    file_type="reference_genome"
                )
                action.log(message_type="reference_genome_path_set", 
                         path=str(self.reference_genome_path) if self.reference_genome_path else None)
            
            # VCF file (optional - may be provided later in fit)
            if self.vcf_path is None and self.vcf_url is not None:
                self._download_or_set_path(
                    url=self.vcf_url, 
                    path_attr="vcf_path",
                    file_type="vcf"
                )
                action.log(message_type="vcf_path_set", 
                         path=str(self.vcf_path) if self.vcf_path else None)
                
            action.log(message_type="file_paths_prepared")
        return self
    
    def model_post_init(self, __context) -> None:
        """Initialize model components once after model creation"""
        # Initialize model and tokenizer if needed
        if self.model_instance is None or self.tokenizer_instance is None:
            eliot.Message.log(message_type="model_loading_start", model_path=self.model_path)
            self.model_instance = AnnotatedModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer_instance = AnnotationTokenizer.from_pretrained(self.tokenizer_path)
            eliot.Message.log(message_type="model_loading_complete")
        
        # Initialize embedder if needed
        if self.embedder_instance is None and self.model_instance is not None and self.tokenizer_instance is not None:
            self.embedder_instance = Embedder(self.model_instance, self.tokenizer_instance)
        
        # Initialize VCF converter if needed
        if self.vcf_converter_instance is None and self.has_required_files:
            self.vcf_converter_instance = VCFConverter(
                gene_annotation_path=str(self.gene_annotation_path),
                reference_genome_path=str(self.reference_genome_path)
            )

    def _download_or_set_path(self, url: str, path_attr: str, file_type: str) -> None:
        """Helper method to download or set a file path based on URL"""
        url_path = Path(url)
        
        # If URL appears to be a local file path that exists
        if not url.startswith(('http://', 'https://', 'ftp://')) and url_path.exists():
            setattr(self, path_attr, url_path)
            return
            
        # Otherwise treat as URL or path that needs downloading
        downloaded_path = download_file(url, self.data_dir, self.force_download)
        if downloaded_path:
            setattr(self, path_attr, downloaded_path)
        else:
            raise FileNotFoundError(f"Failed to download {file_type} file from {url}")
    
    @computed_field
    def has_required_files(self) -> bool:
        """Check if all required reference files exist"""
        return (
            self.gene_annotation_path is not None and self.gene_annotation_path.exists() and
            self.reference_genome_path is not None and self.reference_genome_path.exists()
        )

    def fit(self, vcf_path: Optional[Union[str, Path]] = None, max_variants: Optional[int] = None) -> 'BioFM':
        """
        Process a VCF file and compute embeddings
        
        Args:
            vcf_path: Path to the VCF file to process. If None, will use the already set vcf_path
            max_variants: Maximum number of variants to process from VCF file.
                         If provided, overrides the instance's max_variants value.
            
        Returns:
            self: The fitted estimator (self)
        """
        with eliot.start_action(action_type="biofm_fit") as action:
            # If max_variants is provided, temporarily override the instance's value
            original_max_variants = None
            if max_variants is not None:
                original_max_variants = self.max_variants
                self.max_variants = max_variants
                action.log(message_type="max_variants_override", max_variants=max_variants)
            
            # If vcf_path is provided, update the instance's vcf_path
            if vcf_path is not None:
                vcf_path = Path(vcf_path) if isinstance(vcf_path, str) else vcf_path
                if not vcf_path.exists():
                    action.log(message_type="vcf_file_missing", path=str(vcf_path))
                    raise FileNotFoundError(f"VCF file not found: {vcf_path}")
                self.vcf_path = vcf_path
                action.log(message_type="vcf_path_updated", path=str(self.vcf_path))
            
            # Make sure we have a vcf_path to process
            if self.vcf_path is None:
                # Try to download default VCF example
                action.log(message_type="vcf_path_missing_downloading_default")
                vcf_path = download_file(DEFAULT_VCF_URL, self.data_dir, self.force_download)
                if vcf_path:
                    self.vcf_path = vcf_path
                    action.log(message_type="vcf_path_downloaded", path=str(self.vcf_path))
                else:
                    action.log(message_type="vcf_path_missing")
                    raise ValueError("No VCF file provided. Please provide a vcf_path parameter or set vcf_url")
            
            # Validate VCF file exists
            if not self.vcf_path.exists():
                action.log(message_type="vcf_file_missing", path=str(self.vcf_path))
                raise FileNotFoundError(f"VCF file not found: {self.vcf_path}")
            
            # Ensure all components are initialized
            action.log(message_type="initializing_components")
            if self.model_instance is None or self.tokenizer_instance is None or self.embedder_instance is None or self.vcf_converter_instance is None:
                self.model_post_init(None)
            
            # Count total variants in VCF file before processing
            total_variants = self._count_variants_in_vcf(self.vcf_path)
            variants_to_process = min(total_variants, self.max_variants) if self.max_variants > 0 and total_variants >= 0 else total_variants
            variants_to_skip = max(0, total_variants - variants_to_process) if total_variants >= 0 else 0
            
            action.log(
                message_type="variants_processing_info",
                total_variants=total_variants,
                variants_to_process=variants_to_process,
                variants_to_skip=variants_to_skip,
                max_variants_limit=self.max_variants
            )
            
            # Process VCF and extract embeddings
            action.log(message_type="processing_start", vcf_path=str(self.vcf_path), max_variants=self.max_variants)
            self.dataset = self.vcf_converter_instance.vcf_to_annotated_dataset(
                vcf_path=str(self.vcf_path), 
                max_variants=self.max_variants
            )
            
            # Log the actual number of variants processed
            actual_processed = len(self.dataset) if self.dataset else 0
            action.log(
                message_type="dataset_created", 
                dataset_size=actual_processed,
                expected_size=variants_to_process,
                difference=variants_to_process - actual_processed if variants_to_process >= 0 and actual_processed >= 0 else None
            )
            
            # Get embeddings and ensure they're properly stored
            raw_embeddings = self.embedder_instance.get_dataset_embeddings(self.dataset)
            self.embeddings = raw_embeddings
            
            # Log the shape of the embeddings
            if self.embeddings is not None:
                if isinstance(self.embeddings, dict) and 'embeddings' in self.embeddings:
                    embedding_shape = self.embeddings['embeddings'].shape if hasattr(self.embeddings['embeddings'], 'shape') else None
                    has_labels = 'labels' in self.embeddings
                else:
                    # For backward compatibility with older tuple format
                    if isinstance(self.embeddings, tuple) and len(self.embeddings) >= 1:
                        embedding_shape = self.embeddings[0].shape if hasattr(self.embeddings[0], 'shape') else None
                        # Convert tuple format to dict format for consistency
                        self.embeddings = {
                            'embeddings': self.embeddings[0],
                            'labels': self.embeddings[1] if len(self.embeddings) > 1 else None
                        }
                    else:
                        embedding_shape = None
                        has_labels = False
                    
                action.log(
                    message_type="embeddings_produced", 
                    shape=str(embedding_shape),
                    has_labels=has_labels if 'has_labels' in locals() else None,
                    type=str(type(self.embeddings))
                )
            
            # Restore original max_variants if it was changed
            if original_max_variants is not None:
                self.max_variants = original_max_variants
                action.log(message_type="max_variants_restored", max_variants=original_max_variants)
                
            return self
    
    def _count_variants_in_vcf(self, vcf_path: Path) -> int:
        """
        Count the total number of variants in a VCF file
        
        Args:
            vcf_path: Path to the VCF file
            
        Returns:
            int: Total number of variants in the file
        """
        import gzip
        # Check if the file is gzipped
        is_gzipped = str(vcf_path).endswith('.gz')
        
        # Open the file with appropriate method
        opener = gzip.open if is_gzipped else open
        count = 0
        
        with opener(vcf_path, 'rt') as f:
            for line in f:
                # Skip header lines
                if not line.startswith('#'):
                    count += 1
        
        return count
    
    def predict(self, vcf_path: Optional[Union[str, Path]] = None, max_variants: Optional[int] = None, 
              dataset_name: Optional[str] = None, dataset_subset: Optional[Union[str, DatasetSubset]] = DatasetSubset.EXPRESSION, 
              return_embeddings: bool = True) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Run the BioFM model prediction using this configuration
        
        Args:
            vcf_path: Path to the VCF file to process. If None, will try to use 
                      the already set vcf_path.
            max_variants: Maximum number of variants to process from VCF file.
                         If provided, overrides the instance's max_variants value.
            dataset_name: Optional benchmark dataset name for linear probing (e.g., "m42-health/variant-benchmark")
            dataset_subset: Subset of the benchmark dataset to use. Can be a DatasetSubset enum value
                           or a string matching one of the enum values.
            return_embeddings: Whether to include embeddings in the return value (can be memory intensive)
        
        Returns:
            If dataset_name is None:
                The model's predictions as embeddings dictionary:
                {
                    'embeddings': array of shape (num_variants, 2*embedding_dim),  # Numeric embeddings for each variant
                    'labels': array of shape (num_variants,)  # Present only during supervised embedding extraction
                }
            If dataset_name is provided:
                A dictionary containing:
                {
                    'embeddings': if return_embeddings=True, the embeddings dictionary
                    'linear_probing_results': Results from linear probing including true labels, predictions, and probabilities
                }
        """
        # Store original max_variants to restore later if needed
        original_max_variants = None
        if max_variants is not None:
            original_max_variants = self.max_variants
            self.max_variants = max_variants
        
        # Fit model if necessary
        if vcf_path is not None or self.dataset is None:
            self.fit(vcf_path)
        
        # If no dataset_name is provided, just return the embeddings
        if dataset_name is None:
            result = self.embeddings
        else:
            # Perform linear probing against the benchmark dataset
            with eliot.start_action(action_type="linear_probing") as action:
                # Convert DatasetSubset enum to string if needed
                subset_value = dataset_subset.value if isinstance(dataset_subset, DatasetSubset) else dataset_subset
                
                action.log(message_type="loading_benchmark_dataset", 
                         dataset_name=dataset_name, 
                         dataset_subset=subset_value)
                
                # Import here to avoid circular imports
                from datasets import load_dataset
                
                # Load the benchmark dataset
                dataset_dict = load_dataset(dataset_name, subset_value)
                action.log(message_type="benchmark_dataset_loaded")
                
                # Prepare combined dataset for linear probing
                combined_dataset = {
                    'train': dataset_dict['train'],
                    'test': self.dataset
                }
                
                # Perform linear probing
                action.log(message_type="starting_linear_probing")
                probing_results = self.embedder_instance.linear_probing(
                    combined_dataset,
                    batch_size=32  # Could make this configurable
                )
                action.log(message_type="linear_probing_complete")
                
                # Prepare result dictionary
                result = {'linear_probing_results': probing_results}
                if return_embeddings:
                    result['embeddings'] = self.embeddings
        
        # Restore original max_variants if it was changed
        if original_max_variants is not None:
            self.max_variants = original_max_variants
            
        return result
            
    def transform(self, vcf_path: Optional[Union[str, Path]] = None, max_variants: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Transform a VCF file into embeddings
        
        Args:
            vcf_path: Path to the VCF file to process. If None, will use the already computed embeddings
            max_variants: Maximum number of variants to process from VCF file.
                         If provided, overrides the instance's max_variants value.
            
        Returns:
            The embeddings dictionary:
            {
                'embeddings': array of shape (num_variants, 2*embedding_dim),  # Numeric embeddings for each variant
                'labels': array of shape (num_variants,)  # Present only during supervised embedding extraction
            }
        """
        if vcf_path is not None or self.embeddings is None:
            original_max_variants = None
            if max_variants is not None:
                original_max_variants = self.max_variants
                self.max_variants = max_variants
            
            self.fit(vcf_path)
            
            if original_max_variants is not None:
                self.max_variants = original_max_variants
        
        return self.embeddings
    
    def fit_transform(self, vcf_path: Optional[Union[str, Path]] = None, max_variants: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Fit and transform in one step
        
        Args:
            vcf_path: Path to the VCF file to process
            max_variants: Maximum number of variants to process from VCF file.
                         If provided, overrides the instance's max_variants value.
            
        Returns:
            The embeddings dictionary:
            {
                'embeddings': array of shape (num_variants, 2*embedding_dim),  # Numeric embeddings for each variant
                'labels': array of shape (num_variants,)  # Present only during supervised embedding extraction
            }
        """
        original_max_variants = None
        if max_variants is not None:
            original_max_variants = self.max_variants
            self.max_variants = max_variants
        
        self.fit(vcf_path)
        result = self.embeddings
        
        if original_max_variants is not None:
            self.max_variants = original_max_variants
            
        return result 
        
    def train_classifier(self, 
                        labels: Union[np.ndarray, list], 
                        embeddings: Optional[np.ndarray] = None,
                        test_size: float = 0.2,
                        random_state: int = 42,
                        model_type: str = 'logistic',
                        model_params: Optional[Dict[str, Any]] = None,
                        return_model: bool = False) -> Dict[str, Any]:
        """
        Train a linear classifier on BioFM embeddings
        
        Args:
            labels: Target labels for training the classifier
            embeddings: Optional embeddings to use for training. If None, uses self.embeddings['embeddings']
            test_size: Proportion of the dataset to include in the test split
            random_state: Controls the shuffling applied to the data before applying the split
            model_type: Type of model to train. Currently supported: 'logistic' (logistic regression)
            model_params: Optional parameters to pass to the model constructor
            return_model: Whether to include the trained model in the return value
            
        Returns:
            Dictionary containing:
            {
                'accuracy': Classification accuracy on test set,
                'f1': F1 score on test set,
                'precision': Precision score on test set,
                'recall': Recall score on test set,
                'roc_auc': ROC AUC score on test set (for binary classification),
                'predictions': Predicted labels for test set,
                'true_labels': True labels for test set,
                'probabilities': Predicted probabilities for test set,
                'model': Trained model object (if return_model=True)
            }
        """
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, 
            recall_score, roc_auc_score
        )
        
        with eliot.start_action(action_type="train_classifier") as action:
            # Get embeddings if not provided
            if embeddings is None:
                if self.embeddings is None or 'embeddings' not in self.embeddings:
                    action.log(message_type="embeddings_missing")
                    raise ValueError("No embeddings available. Run fit() or provide embeddings parameter")
                embeddings = self.embeddings['embeddings']
            
            # Validate inputs
            if len(labels) != embeddings.shape[0]:
                action.log(message_type="shape_mismatch", 
                         labels_shape=len(labels), 
                         embeddings_shape=embeddings.shape[0])
                raise ValueError(f"Labels length ({len(labels)}) must match embeddings first dimension ({embeddings.shape[0]})")
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size=test_size, random_state=random_state
            )
            
            action.log(message_type="data_split", 
                     train_size=X_train.shape[0], 
                     test_size=X_test.shape[0])
            
            # Set up model with default parameters if none provided
            if model_params is None:
                model_params = {}
            
            # Create and train model
            if model_type.lower() == 'logistic':
                # Set sensible defaults if not specified
                if 'max_iter' not in model_params:
                    model_params['max_iter'] = 5000
                if 'random_state' not in model_params:
                    model_params['random_state'] = random_state
                
                model = LogisticRegression(**model_params)
            else:
                action.log(message_type="unsupported_model_type", model_type=model_type)
                raise ValueError(f"Unsupported model type: {model_type}. Currently supported: 'logistic'")
            
            action.log(message_type="training_start", model_type=model_type, model_params=model_params)
            model.fit(X_train, y_train)
            action.log(message_type="training_complete")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities (for binary classification)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                # For binary classification, we're interested in the probability of the positive class
                if y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]
            else:
                y_proba = None
            
            # Compute metrics
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'predictions': y_pred,
                'true_labels': y_test,
                'probabilities': y_proba
            }
            
            # Add ROC AUC score for binary classification if we have probabilities
            if y_proba is not None and len(np.unique(y_test)) == 2:
                results['roc_auc'] = roc_auc_score(y_test, y_proba)
            
            # Include the model if requested
            if return_model:
                results['model'] = model
            
            action.log(message_type="evaluation_complete", 
                     accuracy=results['accuracy'], 
                     f1=results['f1'])
            
            return results
            
    def cross_validate_classifier(self, 
                                labels: Union[np.ndarray, list],
                                embeddings: Optional[np.ndarray] = None,
                                n_splits: int = 5,
                                shuffle: bool = True,
                                random_state: int = 42,
                                model_type: str = 'logistic',
                                model_params: Optional[Dict[str, Any]] = None,
                                return_final_model: bool = False) -> Dict[str, Any]:
        """
        Evaluate classifier performance using cross-validation
        
        Args:
            labels: Target labels for training the classifier
            embeddings: Optional embeddings to use for training. If None, uses self.embeddings['embeddings']
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle the data before splitting
            random_state: Controls the shuffling applied to the data before applying the split
            model_type: Type of model to train. Currently supported: 'logistic' (logistic regression)
            model_params: Optional parameters to pass to the model constructor
            return_final_model: Whether to train and return a final model on all data
            
        Returns:
            Dictionary containing:
            {
                'cv_results': Dictionary with cross-validation results:
                    - 'accuracy': List of accuracy scores for each fold
                    - 'f1': List of f1 scores for each fold
                    - 'precision': List of precision scores for each fold
                    - 'recall': List of recall scores for each fold
                    - 'roc_auc': List of ROC AUC scores for each fold (for binary classification)
                'mean_results': Dictionary with mean scores across all folds
                'std_results': Dictionary with standard deviation of scores across all folds
                'final_model': Trained model on all data (if return_final_model=True)
            }
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, 
            recall_score, roc_auc_score
        )
        
        with eliot.start_action(action_type="cross_validate_classifier") as action:
            # Get embeddings if not provided
            if embeddings is None:
                if self.embeddings is None or 'embeddings' not in self.embeddings:
                    action.log(message_type="embeddings_missing")
                    raise ValueError("No embeddings available. Run fit() or provide embeddings parameter")
                embeddings = self.embeddings['embeddings']
            
            # Validate inputs
            if len(labels) != embeddings.shape[0]:
                action.log(message_type="shape_mismatch", 
                         labels_shape=len(labels), 
                         embeddings_shape=embeddings.shape[0])
                raise ValueError(f"Labels length ({len(labels)}) must match embeddings first dimension ({embeddings.shape[0]})")
            
            # Convert labels to numpy array for consistency
            labels = np.array(labels)
            
            # Set up cross-validation
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            
            # Set up model parameters
            if model_params is None:
                model_params = {}
                
            # Initialize results containers
            fold_accuracies = []
            fold_f1s = []
            fold_precisions = []
            fold_recalls = []
            fold_roc_aucs = []
            
            action.log(message_type="cross_validation_start", 
                     n_splits=n_splits, 
                     model_type=model_type)
            
            # Perform cross-validation
            for i, (train_idx, test_idx) in enumerate(cv.split(embeddings, labels)):
                X_train, X_test = embeddings[train_idx], embeddings[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
                
                # Create and train model
                if model_type.lower() == 'logistic':
                    # Set sensible defaults if not specified
                    if 'max_iter' not in model_params:
                        model_params['max_iter'] = 5000
                    if 'random_state' not in model_params:
                        model_params['random_state'] = random_state
                    
                    model = LogisticRegression(**model_params)
                else:
                    action.log(message_type="unsupported_model_type", model_type=model_type)
                    raise ValueError(f"Unsupported model type: {model_type}. Currently supported: 'logistic'")
                
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics for this fold
                fold_accuracies.append(accuracy_score(y_test, y_pred))
                fold_f1s.append(f1_score(y_test, y_pred, average='weighted'))
                fold_precisions.append(precision_score(y_test, y_pred, average='weighted'))
                fold_recalls.append(recall_score(y_test, y_pred, average='weighted'))
                
                # Calculate ROC AUC for binary classification
                if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fold_roc_aucs.append(roc_auc_score(y_test, y_proba))
                    
                action.log(message_type="fold_complete", 
                         fold=i+1, 
                         accuracy=fold_accuracies[-1],
                         f1=fold_f1s[-1])
            
            # Compile cross-validation results
            cv_results = {
                'accuracy': fold_accuracies,
                'f1': fold_f1s,
                'precision': fold_precisions,
                'recall': fold_recalls
            }
            
            if fold_roc_aucs:
                cv_results['roc_auc'] = fold_roc_aucs
            
            # Calculate mean and std of metrics
            mean_results = {metric: np.mean(scores) for metric, scores in cv_results.items()}
            std_results = {metric: np.std(scores) for metric, scores in cv_results.items()}
            
            action.log(message_type="cross_validation_complete",
                     mean_accuracy=mean_results['accuracy'],
                     mean_f1=mean_results['f1'])
            
            # Prepare result dictionary
            results = {
                'cv_results': cv_results,
                'mean_results': mean_results,
                'std_results': std_results
            }
            
            # Train final model on all data if requested
            if return_final_model:
                if model_type.lower() == 'logistic':
                    final_model = LogisticRegression(**model_params)
                    final_model.fit(embeddings, labels)
                    results['final_model'] = final_model
                    action.log(message_type="final_model_trained")
            
            return results
            
    def save_classifier(self, model, output_path: Union[str, Path], 
                       model_metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save a trained classifier model to disk
        
        Args:
            model: Trained classifier model to save
            output_path: Path to save the model to
            model_metadata: Optional metadata to save with the model
            
        Returns:
            Path: Path to the saved model file
        """
        import joblib
        from datetime import datetime
        
        with eliot.start_action(action_type="save_classifier") as action:
            # Convert string path to Path object if needed
            output_path = Path(output_path) if isinstance(output_path, str) else output_path
            
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata
            if model_metadata is None:
                model_metadata = {}
            
            # Add timestamp to metadata
            model_metadata['saved_at'] = datetime.now().isoformat()
            model_metadata['model_type'] = type(model).__name__
            
            # Create a dictionary with both the model and metadata
            save_dict = {
                'model': model,
                'metadata': model_metadata
            }
            
            # Save the model and metadata to disk
            joblib.dump(save_dict, output_path)
            
            action.log(message_type="model_saved", 
                     path=str(output_path), 
                     model_type=model_metadata['model_type'])
            
            return output_path
    
    def load_classifier(self, model_path: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a trained classifier model from disk
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Tuple containing:
            - The loaded model
            - Dictionary of model metadata
        """
        import joblib
        
        with eliot.start_action(action_type="load_classifier") as action:
            # Convert string path to Path object if needed
            model_path = Path(model_path) if isinstance(model_path, str) else model_path
            
            # Check if the model file exists
            if not model_path.exists():
                action.log(message_type="model_file_missing", path=str(model_path))
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the model and metadata from disk
            try:
                loaded_dict = joblib.load(model_path)
                
                # Extract model and metadata
                if isinstance(loaded_dict, dict) and 'model' in loaded_dict and 'metadata' in loaded_dict:
                    model = loaded_dict['model']
                    metadata = loaded_dict['metadata']
                else:
                    # For backward compatibility with models saved without metadata
                    model = loaded_dict
                    metadata = {'model_type': type(model).__name__}
                
                action.log(message_type="model_loaded", 
                         path=str(model_path), 
                         model_type=metadata.get('model_type', type(model).__name__))
                
                return model, metadata
                
            except Exception as e:
                action.log(message_type="model_load_error", 
                         path=str(model_path), 
                         error=str(e))
                raise ValueError(f"Error loading model from {model_path}: {str(e)}")
                
    def predict_with_classifier(self, model, vcf_path: Optional[Union[str, Path]] = None, 
                              max_variants: Optional[int] = None,
                              embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Use a trained classifier to make predictions on variant embeddings
        
        Args:
            model: Trained classifier model
            vcf_path: Path to VCF file to process. If None, will use the already computed embeddings
            max_variants: Maximum number of variants to process from VCF file.
                         If provided, overrides the instance's max_variants value.
            embeddings: Optional embeddings to use for prediction. If None, extracts embeddings
                       from vcf_path or uses self.embeddings['embeddings']
            
        Returns:
            numpy.ndarray: Array of predicted labels
        """
        with eliot.start_action(action_type="predict_with_classifier") as action:
            # Get embeddings for prediction
            if embeddings is None:
                if vcf_path is not None or self.embeddings is None:
                    # Extract embeddings from VCF file
                    result = self.transform(vcf_path, max_variants)
                    embeddings = result['embeddings']
                else:
                    # Use existing embeddings
                    if 'embeddings' not in self.embeddings:
                        action.log(message_type="embeddings_missing")
                        raise ValueError("No 'embeddings' key in self.embeddings dictionary")
                    embeddings = self.embeddings['embeddings']
            
            action.log(message_type="predicting", 
                     embeddings_shape=str(embeddings.shape))
            
            # Make predictions
            predictions = model.predict(embeddings)
            
            # Get prediction probabilities if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(embeddings)
                action.log(message_type="prediction_complete", 
                         predictions_shape=str(predictions.shape),
                         has_probabilities=True)
                
                # For binary classification, return probabilities of the positive class
                if probabilities.shape[1] == 2:
                    return predictions, probabilities[:, 1]
                else:
                    return predictions, probabilities
            else:
                action.log(message_type="prediction_complete", 
                         predictions_shape=str(predictions.shape),
                         has_probabilities=False)
                return predictions
                
    def evaluate_classifier(self, model, labels: Union[np.ndarray, list], 
                          vcf_path: Optional[Union[str, Path]] = None,
                          max_variants: Optional[int] = None,
                          embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate a trained classifier on a new dataset
        
        Args:
            model: Trained classifier model
            labels: True labels for evaluation
            vcf_path: Path to VCF file to process. If None, will use the already computed embeddings
            max_variants: Maximum number of variants to process from VCF file.
                         If provided, overrides the instance's max_variants value.
            embeddings: Optional embeddings to use for evaluation. If None, extracts embeddings
                       from vcf_path or uses self.embeddings['embeddings']
            
        Returns:
            Dictionary containing evaluation metrics:
            {
                'accuracy': Classification accuracy,
                'f1': F1 score,
                'precision': Precision score,
                'recall': Recall score,
                'roc_auc': ROC AUC score (for binary classification),
                'confusion_matrix': Confusion matrix,
                'classification_report': Classification report as string,
                'predictions': Predicted labels,
                'probabilities': Predicted probabilities (if available)
            }
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        import numpy as np
        
        with eliot.start_action(action_type="evaluate_classifier") as action:
            # Get predictions
            prediction_result = self.predict_with_classifier(model, vcf_path, max_variants, embeddings)
            
            # Handle different return types from predict_with_classifier
            if isinstance(prediction_result, tuple):
                predictions, probabilities = prediction_result
            else:
                predictions = prediction_result
                probabilities = None
            
            # Convert labels to numpy array if needed
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            
            # Check if labels and predictions have the same length
            if len(labels) != len(predictions):
                action.log(message_type="shape_mismatch", 
                         labels_shape=len(labels), 
                         predictions_shape=len(predictions))
                raise ValueError(f"Labels length ({len(labels)}) must match predictions length ({len(predictions)})")
            
            # Calculate metrics
            results = {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted'),
                'precision': precision_score(labels, predictions, average='weighted'),
                'recall': recall_score(labels, predictions, average='weighted'),
                'confusion_matrix': confusion_matrix(labels, predictions),
                'classification_report': classification_report(labels, predictions),
                'predictions': predictions
            }
            
            # Add ROC AUC score for binary classification if probabilities are available
            if probabilities is not None and len(np.unique(labels)) == 2:
                results['roc_auc'] = roc_auc_score(labels, probabilities)
                results['probabilities'] = probabilities
            
            action.log(message_type="evaluation_complete", 
                     accuracy=results['accuracy'], 
                     f1=results['f1'])
            
            return results 