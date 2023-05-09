from colorama import Fore, Style


_tqdm_format = "SBRIGHT{desc}RESET: HI{percentage:3.0f}%RESET {n_fmt}/{total_fmt} [{elapsed}<HI{remaining}RESET, {rate_fmt}]"
tqdm_format = _tqdm_format \
    .replace("HI", Fore.CYAN) \
    .replace("SBRIGHT", Style.BRIGHT) \
    .replace("RESET", Style.RESET_ALL)
