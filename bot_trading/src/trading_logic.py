class TradingLogic:
    def __init__(self, config):
        self.grid_step = config.get("grid_step", 50)
        self.max_positions = config.get("max_positions", 3)
        self.tp_target = config.get("take_profit", 30000)  # dalam poin harga atau USD
        self.sl_limit = config.get("stop_loss", 15000)
        self.positions = []  # List menyimpan posisi aktif: dict dengan tipe, harga entry, TP, SL

    def update_positions(self, current_price):
        """
        Cek posisi aktif untuk exit strategi (TP/SL)
        Kembalikan list posisi yang perlu ditutup
        """
        close_positions = []
        for pos in self.positions:
            if pos["type"] == "BUY":
                if current_price >= pos["entry_price"] + self.tp_target:
                    close_positions.append(pos)
                elif current_price <= pos["entry_price"] - self.sl_limit:
                    close_positions.append(pos)
            elif pos["type"] == "SELL":
                if current_price <= pos["entry_price"] - self.tp_target:
                    close_positions.append(pos)
                elif current_price >= pos["entry_price"] + self.sl_limit:
                    close_positions.append(pos)
        # Hapus posisi yang ditutup dari active list
        for pos in close_positions:
            self.positions.remove(pos)
        return close_positions

    def can_open_position(self, prediction, confidence):
        """
        Logika open posisi baru:
        - Confidence minimal threshold 0.7 (bisa parameter)
        - Posisi maximal belum terlewati
        """
        if confidence < 0.7:
            return False
        if len(self.positions) >= self.max_positions:
            return False
        # Bisa tambah logika filter lain (misal arah trend EMA)
        return True

    def open_position(self, prediction, current_price, lot_size):
        """
        Buat dict posisi baru dengan parameter entry price, tipe, TP/SL
        """
        pos = {
            "type": prediction,
            "entry_price": current_price,
            "tp": self.tp_target,
            "sl": self.sl_limit,
            "lot": lot_size
        }
        self.positions.append(pos)
        return pos
