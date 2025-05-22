from pathlib import Path
from typing import Dict, Optional, ClassVar, Any, Union, List, Tuple
import numpy as np
from pydantic import BaseModel, Field, model_validator, computed_field
import torch
from datasets.arrow_dataset import Dataset
import eliot
from datetime import datetime

from biomodels_mcp.config import (
    MODEL_PATH, TOKENIZER_PATH, DEFAULT_DATA_DIR, 
    LOGS_DIR,
    # Direct URL constants instead of FILE_SPECS
    DEFAULT_GENE_ANNOTATION_URL,
    DEFAULT_REFERENCE_GENOME_URL,
    DEFAULT_VCF_URL
)
from biomodels_mcp.downloads import (
    download_file,
    get_final_filename
)
from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter


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
    embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = Field(default=None, exclude=True)
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

    def _ensure_model_loaded(self) -> None:
        """Ensure model and tokenizer are loaded"""
        if self.model_instance is None or self.tokenizer_instance is None:
            # We'll log this directly rather than creating a new action
            eliot.Message.log(message_type="model_loading_start", model_path=self.model_path)
            self.model_instance = AnnotatedModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer_instance = AnnotationTokenizer.from_pretrained(self.tokenizer_path)
            eliot.Message.log(message_type="model_loading_complete")
    
    def _ensure_converter_ready(self) -> None:
        """Ensure VCF converter is initialized"""
        if not self.has_required_files:
            raise ValueError("Required reference files are missing. Call prepare_file_paths first.")
            
        if self.vcf_converter_instance is None:
            self.vcf_converter_instance = VCFConverter(
                gene_annotation_path=str(self.gene_annotation_path),
                reference_genome_path=str(self.reference_genome_path)
            )
    
    def _ensure_embedder_ready(self) -> None:
        """Ensure embedder is initialized"""
        self._ensure_model_loaded()
        if self.embedder_instance is None:
            self.embedder_instance = Embedder(self.model_instance, self.tokenizer_instance)
    
    def fit(self, vcf_path: Optional[Union[str, Path]] = None) -> 'BioFM':
        """
        Process a VCF file and compute embeddings
        
        Args:
            vcf_path: Path to the VCF file to process. If None, will use the already set vcf_path
            
        Returns:
            self: The fitted estimator (self)
        """
        with eliot.start_action(action_type="biofm_fit") as action:
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
            self._ensure_model_loaded()
            self._ensure_converter_ready()
            self._ensure_embedder_ready()
            
            # Process VCF and extract embeddings
            action.log(message_type="processing_start", vcf_path=str(self.vcf_path))
            self.dataset = self.vcf_converter_instance.vcf_to_annotated_dataset(
                vcf_path=str(self.vcf_path), 
                max_variants=self.max_variants
            )
            action.log(message_type="dataset_created", dataset_size=len(self.dataset) if self.dataset else 0)
            
            self.embeddings = self.embedder_instance.get_dataset_embeddings(self.dataset)
            
            # Log the shape of the embeddings
            if self.embeddings is not None:
                embedding_shape = self.embeddings[0].shape if hasattr(self.embeddings[0], 'shape') else None
                action.log(message_type="embeddings_produced", 
                         shape=str(embedding_shape),
                         type=str(type(self.embeddings)))
                
            return self
    
    def predict(self, vcf_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the BioFM model prediction using this configuration
        
        Args:
            vcf_path: Path to the VCF file to process. If None, will try to use 
                      the already set vcf_path.
        
        Returns:
            The model's predictions as embeddings tuple (two numpy arrays)
        """
        try:
            if vcf_path is not None or self.dataset is None:
                self.fit(vcf_path)
            
            # Return the embeddings as predictions
            return self.embeddings
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
            
    def transform(self, vcf_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a VCF file into embeddings
        
        Args:
            vcf_path: Path to the VCF file to process. If None, will use the already computed embeddings
            
        Returns:
            The embeddings as a tuple of two numpy arrays
        """
        if vcf_path is not None or self.embeddings is None:
            self.fit(vcf_path)
        
        return self.embeddings
    
    def fit_transform(self, vcf_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step
        
        Args:
            vcf_path: Path to the VCF file to process
            
        Returns:
            The embeddings as a tuple of two numpy arrays
        """
        self.fit(vcf_path)
        return self.embeddings 