from datasets import BridgeRuler

bridge = BridgeRuler()
for x, y in bridge:
    # print(x.shape, x.dtype, y.shape, y.dtype)
    print(y.max(), y.dtype)
