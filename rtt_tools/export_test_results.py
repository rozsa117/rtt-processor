import json
from builtins import filter
from math import ceil

from rtt_tools import dump_data

DIEHARDER_MAX_ALLOWED_TO_PASS_8GiB = 23
DIEHARDER_MAX_ALLOWED_TO_PASS_100MiB = 19
DIEHARDER_MAX_ALLOWED_TO_PASS_10MiB = 17

NIST_MAX_ALLOWED_TO_PASS = 12

ALPHABIT_MAX_ALLOWED_TO_PASS = 2

RABBIT_MAX_ALLOWED_TO_PASS = 13

# no test run fo 10 and 100 MiB
CRUSH_MAX_ALLOWED_TO_PASS = 28

# no tests run for 10MiB
SMALL_CRUSH_MAX_ALLOWED_TO_PASS_8GiB = 7
SMALL_CRUSH_MAX_ALLOWED_TO_PASS_100MiB = 3

BOOLTEST_MAX_ALLOWED_TO_PASS = 34

FOLDER_NAME = "final_data/"

NIST_BATTERY_NAME = 'NIST'
DIEHARDER_BATTERY_NAME = 'Dieharder'
TESTU01_BATTERY_NAME = 'TestU01'
BOOLTEST_BATTERY_NAME = 'booltest'
BATTERY_NAMES = [NIST_BATTERY_NAME, DIEHARDER_BATTERY_NAME, TESTU01_BATTERY_NAME, BOOLTEST_BATTERY_NAME]

OSIZE_8GB = '8GiB'
OSIZE_100MB = '100MiB'
OSIZE_10MB = '10MiB'

RANGE_OF_ROUNDS_PREFIX = "range_of_rounds_"

PRNGS = ('testu01-ulcg', 'testu01-umrg', 'testu01-uxorshift', 'std_lcg', 'std_subtract_with_carry', 'std_mersenne_twister')


def get_data_to_parse(exp_data) -> dict:
    needed_data = dict()
    function_names = set()
    battery_names = set()
    data_sizes = set()
    strategies = set()
    for experiment in exp_data.values():
        if not experiment['fnc'] in PRNGS:
            function_names.add(experiment['fnc'])
            data_sizes.add(experiment['osize'])
            strategies.add(experiment['meth'])
            for battery in experiment['batteries']:
                battery_names.add(battery['name'])

    needed_data['functions'] = function_names
    needed_data['batteries'] = battery_names
    needed_data['data_sizes'] = data_sizes
    needed_data['strategies'] = strategies
    return needed_data


def has_battery_detected_bias(battery, osize) -> bool:
    if NIST_BATTERY_NAME in battery['name']:
        return battery['passed'] <= NIST_MAX_ALLOWED_TO_PASS
    elif DIEHARDER_BATTERY_NAME in battery['name']:
        return (osize == OSIZE_8GB and battery['passed'] <= DIEHARDER_MAX_ALLOWED_TO_PASS_8GiB) or (
                osize == OSIZE_100MB and battery['passed'] <= DIEHARDER_MAX_ALLOWED_TO_PASS_100MiB) or (
                       osize == OSIZE_10MB and battery['passed'] <= DIEHARDER_MAX_ALLOWED_TO_PASS_10MiB)
    elif BOOLTEST_BATTERY_NAME in battery['name']:
        return battery['passed'] <= BOOLTEST_MAX_ALLOWED_TO_PASS
    elif battery['name'] == 'TestU01 Rabbit':
        return battery['passed'] <= RABBIT_MAX_ALLOWED_TO_PASS
    elif battery['name'] == 'TestU01 Small Crush':
        return (osize == OSIZE_8GB and battery['passed'] <= SMALL_CRUSH_MAX_ALLOWED_TO_PASS_8GiB) or (
                osize == OSIZE_100MB and battery['passed'] <= SMALL_CRUSH_MAX_ALLOWED_TO_PASS_100MiB)
    elif battery['name'] == 'TestU01 Crush':
        return battery['passed'] <= CRUSH_MAX_ALLOWED_TO_PASS
    elif 'Alphabit' in battery['name']:  # TestU01 Alphabit and TestU01 Block Alphabit
        return battery['passed'] <= ALPHABIT_MAX_ALLOWED_TO_PASS


def has_battery_detected_previous_tested_round(previous_round, experiments, battery_name) -> bool:
    for experiment in experiments:
        if experiment['round'] == previous_round:
            for battery in experiment['batteries']:
                if battery['name'] == battery_name:
                    return has_battery_detected_bias(battery, experiment['osize'])


def replace_if_not_false_positive(previous_result, new_result, rounds, experiments, battery_name) -> int:
    if previous_result == new_result or rounds.index(new_result) == 0:
        return new_result
    else:
        return new_result if has_battery_detected_previous_tested_round(rounds[rounds.index(new_result) - 1],
                                                                        experiments,
                                                                        battery_name) else previous_result


def get_tested_rounds_for_function(experiments) -> list:
    rounds = set()
    for experiment in experiments:
        rounds.add(experiment['round'])
    return list(rounds)


def get_number_of_passed_tests_for_round_and_battery(selected_experiments, battery_name, selected_round, seed) -> int:
    passed_tests = 0
    for experiment in selected_experiments:
        if experiment['round'] == selected_round and experiment['seed'] == seed:
            for battery in experiment['batteries']:
                if battery_name in battery['name']:
                    passed_tests += battery['passed']

    return passed_tests


def should_test_round(selected_experiments, battery_name, lower_tested_round, higher_tested_rounds, seed) -> bool:
    passed_tests_for_lower_round = get_number_of_passed_tests_for_round_and_battery(selected_experiments, battery_name,
                                                                                    lower_tested_round,
                                                                                    seed)
    passed_tests_for_higher_round = get_number_of_passed_tests_for_round_and_battery(selected_experiments, battery_name,
                                                                                     higher_tested_rounds, seed)
    return abs(passed_tests_for_higher_round - passed_tests_for_lower_round) > 1


def get_lower_rounds(selected_experiments, highest_detected_round, rounds, battery_name, seed) -> list:
    if highest_detected_round == 0:
        return []
    lower_rounds = []
    round_to_add = highest_detected_round
    while rounds.index(round_to_add) > 0:
        round_to_add = rounds[rounds.index(round_to_add) - 1]
        if rounds.index(round_to_add) >= rounds.index(highest_detected_round) - 2:
            lower_rounds.append(round_to_add)
        elif should_test_round(selected_experiments, battery_name, round_to_add, rounds[rounds.index(round_to_add) + 1],
                               seed):
            lower_rounds.append(round_to_add)
        else:
            return lower_rounds
    return lower_rounds


def get_higher_rounds(selected_experiments, highest_detected_round, rounds, battery_name, seed) -> list:
    if rounds.index(highest_detected_round) + 1 == len(rounds):
        return []
    higher_rounds = []
    round_to_add = highest_detected_round
    while rounds.index(round_to_add) + 1 < len(rounds):
        round_to_add = rounds[rounds.index(round_to_add) + 1]
        if rounds.index(round_to_add) <= rounds.index(highest_detected_round) + 2:
            higher_rounds.append(round_to_add)
        elif should_test_round(selected_experiments, battery_name, round_to_add, rounds[rounds.index(round_to_add) - 1],
                               seed):
            higher_rounds.append(round_to_add)
        else:
            return higher_rounds
    return higher_rounds


def get_range_of_rounds_for_battery_per_seed(selected_experiments, highest_detected_round, rounds, battery_name,
                                             seed) -> list:
    if highest_detected_round is None:
        return sorted(rounds)[:2]
    else:
        range_of_rounds = list()
        range_of_rounds.append(highest_detected_round)
        range_of_rounds.extend(
            get_lower_rounds(selected_experiments, highest_detected_round, rounds, battery_name, seed))
        range_of_rounds.extend(
            get_higher_rounds(selected_experiments, highest_detected_round, rounds, battery_name, seed))
        return sorted(range_of_rounds, key=lambda r: r)


def map_to_dict_of_max(results_for_each_seed) -> list:
    max_results = list()
    for value in results_for_each_seed.values():
        max_results.append(value[-1])
    return max_results


def map_to_dict_of_min(results_for_each_seed) -> list:
    min_values = list()
    for value in results_for_each_seed.values():
        min_values.append(value[0])
    return min_values


def get_range_of_rounds_for_battery(selected_experiments, highest_detected_round, rounds, battery_name, seeds) -> list:
    results_for_each_seed = dict()
    for seed in seeds:
        results_for_each_seed[seed] = sorted(get_range_of_rounds_for_battery_per_seed(selected_experiments,
                                                                                      highest_detected_round, rounds,
                                                                                      battery_name, seed))

    max_round = get_result(map_to_dict_of_max(results_for_each_seed), ceil(len(seeds) * 2 / 3))
    min_round = get_result(map_to_dict_of_min(results_for_each_seed), ceil(len(seeds) * 2 / 3))
    return list(x for x in rounds if min_round <= x <= max_round)


def get_max_round_where_bias_was_detected_for_given_seed(selected_experiments) -> dict:
    result_nist = None
    result_dieharder = None
    result_testu01 = None
    result_booltest = None
    batteries_not_test_further = set()
    tested_rounds = sorted(get_tested_rounds_for_function(selected_experiments))
    for experiment in selected_experiments:
        for battery in experiment['batteries']:
            if battery['total'] > 0:
                if has_battery_detected_bias(battery, experiment['osize']) \
                        and battery['name'] not in batteries_not_test_further:
                    if NIST_BATTERY_NAME in battery['name']:
                        result_nist = replace_if_not_false_positive(result_nist, experiment['round'], tested_rounds,
                                                                    selected_experiments,
                                                                    battery['name'])
                    elif DIEHARDER_BATTERY_NAME in battery['name']:
                        result_dieharder = replace_if_not_false_positive(result_dieharder, experiment['round'],
                                                                         tested_rounds,
                                                                         selected_experiments,
                                                                         battery['name'])
                    elif BOOLTEST_BATTERY_NAME in battery['name']:
                        result_booltest = replace_if_not_false_positive(result_booltest, experiment['round'],
                                                                        tested_rounds,
                                                                        selected_experiments,
                                                                        battery['name'])
                    elif TESTU01_BATTERY_NAME in battery['name']:
                        result_testu01 = replace_if_not_false_positive(result_testu01, experiment['round'],
                                                                       tested_rounds,
                                                                       selected_experiments,
                                                                       battery['name'])
                else:
                    batteries_not_test_further.add(battery['name'])

    detected_rounds_per_battery = dict()
    detected_rounds_per_battery[NIST_BATTERY_NAME] = result_nist
    detected_rounds_per_battery[DIEHARDER_BATTERY_NAME] = result_dieharder
    detected_rounds_per_battery[TESTU01_BATTERY_NAME] = result_testu01
    detected_rounds_per_battery[BOOLTEST_BATTERY_NAME] = result_booltest
    return detected_rounds_per_battery


def get_result(results, needed_to_consider_valid_value) -> int:
    result_set = set(results)
    for result in result_set:
        if results.count(result) >= needed_to_consider_valid_value:
            return result
    max_occurrence = max(map(results.count, results))
    return min(filter(lambda r: results.count(r) == max_occurrence, results))


def merge_results(results_for_each_seed, seeds) -> dict:
    needed_to_consider_valid_value = ceil(len(seeds) * 2 / 3)
    result_nist = list()
    result_dieharder = list()
    result_testu01 = list()
    result_booltest = list()
    for seed in seeds:
        result_nist.append(results_for_each_seed[seed][NIST_BATTERY_NAME])
        result_dieharder.append(results_for_each_seed[seed][DIEHARDER_BATTERY_NAME])
        result_testu01.append(results_for_each_seed[seed][TESTU01_BATTERY_NAME])
        result_booltest.append(results_for_each_seed[seed][BOOLTEST_BATTERY_NAME])
    result = dict()
    result[NIST_BATTERY_NAME] = get_result(result_nist, needed_to_consider_valid_value)
    result[DIEHARDER_BATTERY_NAME] = get_result(result_dieharder, needed_to_consider_valid_value)
    result[TESTU01_BATTERY_NAME] = get_result(result_testu01, needed_to_consider_valid_value)
    result[BOOLTEST_BATTERY_NAME] = get_result(result_booltest, needed_to_consider_valid_value)
    return result


def write_results_to_file(prefix, osize, meth, results, fnc_name):
    for battery_name in BATTERY_NAMES:
        file = open(FOLDER_NAME + prefix + battery_name + "_" + osize + "_" + meth + ".json", "r")
        json_data = json.load(file)
        file.close()
        file = open(FOLDER_NAME + prefix + battery_name + "_" + osize + "_" + meth + ".json", "w")
        json_data[fnc_name] = results[battery_name]
        json.dump(json_data, file)
        file.close()


def get_max_round_where_bias_was_detected(experiments):
    experiments = sorted(experiments, key=lambda e: e['round'])
    fnc_name = experiments[0]['fnc']
    osize = experiments[0]['osize']
    strategy = experiments[0]['meth']
    results_for_each_seed = dict()
    seeds = set()
    tested_rounds = sorted(get_tested_rounds_for_function(experiments))
    for experiment in experiments:
        seeds.add(experiment['seed'])
    for seed in seeds:
        selected_experiments = []
        for experiment in experiments:
            if experiment['seed'] == seed:
                selected_experiments.append(experiment)
        returned_value = get_max_round_where_bias_was_detected_for_given_seed(selected_experiments)
        results_for_each_seed[seed] = returned_value
    results = merge_results(results_for_each_seed, seeds)
    range_of_rounds = dict()
    range_of_rounds[NIST_BATTERY_NAME] = get_range_of_rounds_for_battery(experiments, results[NIST_BATTERY_NAME],
                                                                         tested_rounds,
                                                                         NIST_BATTERY_NAME, seeds)
    range_of_rounds[DIEHARDER_BATTERY_NAME] = get_range_of_rounds_for_battery(experiments,
                                                                              results[DIEHARDER_BATTERY_NAME],
                                                                              tested_rounds,
                                                                              DIEHARDER_BATTERY_NAME, seeds)
    range_of_rounds[TESTU01_BATTERY_NAME] = get_range_of_rounds_for_battery(experiments, results[TESTU01_BATTERY_NAME],
                                                                            tested_rounds,
                                                                            TESTU01_BATTERY_NAME, seeds)
    range_of_rounds[BOOLTEST_BATTERY_NAME] = get_range_of_rounds_for_battery(experiments,
                                                                             results[BOOLTEST_BATTERY_NAME],
                                                                             tested_rounds,
                                                                             BOOLTEST_BATTERY_NAME, seeds)
    write_results_to_file("", osize, strategy, results, fnc_name)
    write_results_to_file(RANGE_OF_ROUNDS_PREFIX, osize, strategy, range_of_rounds, fnc_name)


def parse_experiments(exp_data, data_sizes, strategies, fnc_names):
    for fnc_name in fnc_names:
        for data_size in data_sizes:
            for strategy in strategies:
                selected_experiments = []
                for experiment in exp_data.values():
                    if experiment['fnc'] == fnc_name and experiment['osize'] == data_size \
                            and experiment['meth'] == strategy:
                        selected_experiments.append(experiment)
                if len(selected_experiments) > 0:
                    get_max_round_where_bias_was_detected(selected_experiments)


def create_json_files(data_sizes, strategies):
    for data_size in data_sizes:
        for strategy in strategies:
            for battery_name in BATTERY_NAMES:
                file = open(
                    FOLDER_NAME + RANGE_OF_ROUNDS_PREFIX + battery_name + "_" + data_size + "_" + strategy + ".json",
                    "w")
                json.dump(dict(), file)
                file.close()
                file = open(FOLDER_NAME + battery_name + "_" + data_size + "_" + strategy + ".json", "w")
                json.dump(dict(), file)
                file.close()


def main():
    exp_data = dump_data.main()
    data = get_data_to_parse(exp_data)
    data_sizes = data['data_sizes']
    strategies = data['strategies']
    fnc_names = data['functions']
    create_json_files(data_sizes, strategies)
    print(len(fnc_names))
    parse_experiments(exp_data, data_sizes, strategies, fnc_names)


if __name__ == "__main__":
    main()
