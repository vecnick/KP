import logging
from pathlib import Path
from ndvi_processor2 import NDVIProcessor2

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ndvi_processor/output/processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Создаем директории
        output_dir = Path("ndvi_processor/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting NDVI processing...")
        
        # Инициализируем процессор
        processor = NDVIProcessor2(
            b4_dir="ndvi_processor/b4",
            b5_dir="ndvi_processor/b5",
            output_dir="ndvi_processor/output",
            chunk_size=256
        )
        
        # Запускаем обработку
        processor.process()
        
        logger.info("NDVI processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 