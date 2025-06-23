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
        
        logger.info("Starting NDVI processing for existing scene...")
        
        # Инициализируем процессор
        processor = NDVIProcessor2(
            b4_dir="b4",  # Путь к существующим данным B4
            b5_dir="b5",  # Путь к существующим данным B5
            output_dir="ndvi_processor/output",
            chunk_size=512
        )
        
        # Запускаем обработку
        processor.process()
        
        logger.info("NDVI processing completed successfully")
        
        # Выводим информацию о графе задач
        task_graph_file = output_dir / 'task_graph.json'
        if task_graph_file.exists():
            logger.info(f"Task graph saved to {task_graph_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 