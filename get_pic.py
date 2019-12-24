from icrawler.builtin import GoogleImageCrawler

#crawler = GoogleImageCrawler(storage={'root_dir':'dog_images'})
#crawler.crawl(keyword='犬', max_num = 100)
crawler = GoogleImageCrawler(storage={'root_dir':'./original_data/maki'})
crawler.crawl(keyword='真木よう子', max_num=1000)
