import csv
import sys

def process_trades(trades):
    pairs = []
    open_trades = {}
    for row in trades:
        time, symbol, side, price, quantity = row
        time, price = int(time), float(price)
        quantity = min(int(quantity), sys.float_info.max)  # Ensure quantity is within float range
        if symbol not in open_trades:
            open_trades[symbol] = []

        if side == 'B':
            open_trades[symbol].append((time, quantity, price))
        elif side == 'S':
            close_quantity = quantity
            while close_quantity > 0 and len(open_trades[symbol])>0:

                open_time, open_quantity, open_price = open_trades[symbol][0]
                paired_quantity = min(open_quantity, close_quantity)
                pnl = paired_quantity * (price - open_price)

                pairs.append((open_time, time, symbol, paired_quantity, pnl, 'B', 'S', open_price, price))

                if open_quantity == close_quantity:
                    open_trades[symbol] = open_trades[symbol][1:]
                elif open_quantity > close_quantity:
                    open_trades[symbol][0] = (open_time, open_quantity - close_quantity, open_price)
                else:
                    open_trades[symbol] = open_trades[symbol][1:]
                    
                close_quantity -= paired_quantity
            if close_quantity >0:
                open_trades[symbol].append((time, -1*close_quantity, price))


    return pairs

def main():
    if len(sys.argv) != 2:
        print("Usage: python your_program.py [trade file]")
        return

    trade_file = sys.argv[1]

    with open(trade_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        trades = [row for row in reader]

    paired_trades = process_trades(trades)

    for trade in sorted(paired_trades, key=lambda x: x[1]):
        print(','.join(map(str, trade)))

    total_pnl = sum(trade[4] for trade in paired_trades)
    print(f"Total PNL: {total_pnl:.2f}")

if __name__ == "__main__":
    main()
