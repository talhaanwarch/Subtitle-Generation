from utils.logging_utils import get_logger

logger = get_logger(__name__)


def check_gpu_availability() -> bool:
    """
    Check if GPU is available using both PyTorch and ONNX Runtime.

    Returns:
        True if GPU is available, False otherwise
    """
    # Try PyTorch first
    try:
        import torch

        if torch.cuda.is_available():
            # logger.info(f"GPU detected via PyTorch: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        logger.debug("PyTorch not available for GPU detection")
    except Exception as e:
        logger.debug(f"PyTorch GPU detection failed: {e}")

    # Try ONNX Runtime as fallback
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            logger.info("GPU detected via ONNX Runtime CUDA provider")
            return True
        elif "ROCMExecutionProvider" in providers:
            logger.info("GPU detected via ONNX Runtime ROCm provider")
            return True
    except ImportError:
        logger.debug("ONNX Runtime not available for GPU detection")
    except Exception as e:
        logger.debug(f"ONNX Runtime GPU detection failed: {e}")

    logger.info("No GPU detected, will use CPU")
    return False
