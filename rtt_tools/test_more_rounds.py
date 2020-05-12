from rtt_tools.export_test_results import *

CTR = 'ctr'
HW = 'hw'
SAC = 'sac'


def test_strategy(file_name):
    print("Testing file - " + file_name)
    with open(FOLDER_NAME + file_name) as f:
        sec_margin = json.load(f)
        with open(FOLDER_NAME + RANGE_OF_ROUNDS_PREFIX + file_name) as f2:
            rounds_range = json.load(f2)
            for key in sec_margin:
                round = sec_margin[key]
                rounds = rounds_range[key]
                if round is None:
                    if not (rounds[0] == 0 or rounds[0] == 1):
                        print("Crypto primitive '" + key + "' has highest detected round '" + str(
                            round) + "' recommended testing range is: " + str(rounds))
                elif (rounds.index(round) == 0 and (round != 1 and round != 0)) or rounds.index(round) == len(
                        rounds) - 1:
                    print("Crypto primitive '" + key + "' has highest detected round '" + str(
                        round) + "' recommended testing range is: " + str(rounds))


def test_size(prefix):
    test_strategy(prefix + "_" + CTR + ".json")
    test_strategy(prefix + "_" + HW + ".json")
    test_strategy(prefix + "_" + SAC + ".json")


def test_battery(function_name):
    test_size(function_name + "_" + OSIZE_10MB)
    test_size(function_name + "_" + OSIZE_100MB)


def main():
    test_battery(BOOLTEST_BATTERY_NAME)
    test_battery(DIEHARDER_BATTERY_NAME)
    test_battery(NIST_BATTERY_NAME)
    test_battery(TESTU01_BATTERY_NAME)


if __name__ == "__main__":
    main()
