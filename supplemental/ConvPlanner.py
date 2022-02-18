
def print_and_apply_conv(in_hw, pad_hw, filter_hw, stride_hw, out_ch):
    out_hw = apply_conv(in_hw, pad_hw, filter_hw, stride_hw)
    total_elems = out_hw[0] * out_hw[1] * out_ch
    print_str = f"VOLUME: ({out_hw[0]}, {out_hw[1]}, {out_ch}) [{total_elems} total elements] - after conv with F {filter_hw}, S {stride_hw}, P {pad_hw}, C {out_ch}"
    print(print_str)
    return out_hw


def apply_conv(in_hw, pad_hw, filter_hw, stride_hw):
    in_h, in_w = in_hw
    pad_h, pad_w = pad_hw
    filter_h, filter_w = filter_hw
    stride_h, stride_w = stride_hw

    out_h = int((in_h + pad_h - filter_h)/stride_h + 1)
    out_w = int((in_w + pad_w - filter_w)/stride_w + 1)

    return out_h, out_w


def print_and_apply_upconv(in_hw, pad_in_hw, pad_out_hw, filter_hw, stride_hw, out_ch):
    out_hw = apply_upconv(in_hw, pad_in_hw, pad_out_hw, filter_hw, stride_hw)
    total_elems = out_hw[0] * out_hw[1] * out_ch
    print_str = f"VOLUME: ({out_hw[0]}, {out_hw[1]}, {out_ch}) [{total_elems} total elements] - after upconv with F {filter_hw}, S {stride_hw}, P(in) {pad_in_hw}, P(out) {pad_out_hw}, C {out_ch}"
    print(print_str)
    return out_hw


def apply_upconv(in_hw, pad_in_hw, pad_out_hw, filter_hw, stride_hw):
    in_h, in_w = in_hw
    pad_in_h, pad_in_w = pad_in_hw
    pad_out_h, pad_out_w = pad_out_hw
    filter_h, filter_w = filter_hw
    stride_h, stride_w = stride_hw

    out_h = (in_h - 1)*stride_h - 2*pad_in_h + pad_out_h + filter_h
    out_w = (in_w - 1) * stride_w - 2 * pad_in_w + pad_out_w + filter_w

    return out_h, out_w


def main():
    start_hw = 375, 1242
    print(f"VOLUME: ({start_hw[0]}, {start_hw[1]}, 3) - input")

    convs = [
        ((0, 0), (5, 5), (2, 2), 4),
        ((0, 0), (5, 5), (2, 2), 4),
        ((0, 0), (5, 5), (2, 2), 8),

        # there are now 4 scales
        ((0, 0), (3, 7), (2, 3), 16),
        ((0, 0), (3, 7), (2, 2), 32),
        ((0, 0), (3, 7), (1, 1), 4)
    ]

    upconvs = [
        ((0, 0), (0, 0), (1, 1), (1, 1), 4),
        ((0, 0), (0, 0), (3, 7), (1, 1), 32),
        ((0, 0), (0, 0), (3, 7), (2, 2), 16),

        # ((0, 0), (0, 0), (3, 5), (2, 2), 4),
        # ((0, 0), (0, 0), (3, 5), (2, 2), 8),
        # ((0, 0), (0, 1), (3, 8), (1, 1), 16),

        # at this point we can use skip connections
        ((0, 0), (1, 1), (3, 7), (2, 3), 8),
        ((0, 0), (0, 1), (5, 5), (2, 2), 4),
        ((0, 0), (1, 0), (5, 5), (2, 2), 4),
        ((0, 0), (0, 1), (5, 5), (2, 2), 1)
    ]

    cur_hw = start_hw
    for conv in convs:
       pad_hw, filter_hw, stride_hw, out_ch = conv
       cur_hw = print_and_apply_conv(cur_hw, pad_hw, filter_hw, stride_hw, out_ch)

    cur_hw = 8, 16
    in_ch = 1
    print(f"\nUPCONV PART - starts with a ({cur_hw[0]}, {cur_hw[1]}, {in_ch}) volume [{in_ch * cur_hw[0] * cur_hw[1]} total elements]")

    for upconv in upconvs:
        pad_in_hw, pad_out_hw, filter_hw, stride_hw, out_ch = upconv
        cur_hw = print_and_apply_upconv(cur_hw, pad_in_hw, pad_out_hw, filter_hw, stride_hw, out_ch)


if __name__ == "__main__":
    main()