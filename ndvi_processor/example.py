from ndvi_processor import NDVIProcessor

def main():
    # Создаем процессор NDVI
    processor = NDVIProcessor(
        b4_dir="b4",  # директория с файлами B4
        b5_dir="b5",  # директория с файлами B5
        output_dir="output"  # директория для результатов
    )
    
    # Запускаем полный процесс обработки
    processor.process()

if __name__ == "__main__":
    main() 