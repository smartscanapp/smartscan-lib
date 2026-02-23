from tqdm import tqdm
from smartscan.processor import ProcessorListener
from smartscan.processor.types import Input, Output

class ProgressBarIndexerListener(ProcessorListener[Input, Output]):
    def __init__(self):
        self.progress_bar = tqdm(total=100, desc="Indexing")

    async def on_progress(self, progress):
        self.progress_bar.n = int(progress * 100)
        self.progress_bar.refresh()
        
    async def on_fail(self, result):
        self.progress_bar.close()
        print(result.error)

    async def on_error(self, e, item):
        print(e)
    
    async def on_complete(self, result):
        self.progress_bar.close()
        print(f"Results: {result.total_processed} | Time elapsed: {result.time_elapsed:.4f}s")

