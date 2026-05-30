class ImageCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size

    def get(self, path):
        if path in self.cache:
            if path in self.access_order:
                self.access_order.remove(path)
            self.access_order.append(path)
            return self.cache[path]
        return None

    def add(self, path, pixmap):
        if path in self.cache:
            if path in self.access_order:
                self.access_order.remove(path)
            self.access_order.append(path)
            self.cache[path] = pixmap
            return

        while len(self.cache) >= self.max_size and self.access_order:
            oldest = self.access_order.pop(0)
            if oldest in self.cache:
                del self.cache[oldest]

        self.cache[path] = pixmap
        self.access_order.append(path)

    def remove(self, path):
        if path in self.cache:
            del self.cache[path]
        if path in self.access_order:
            self.access_order.remove(path)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()

    def __contains__(self, path):
        return path in self.cache

    def __len__(self):
        return len(self.cache)
