import time
import traceback
import os
from dotenv import load_dotenv
from gpt4all import GPT4All
import wolframalpha
import redis
import colorama as color
from tqdm import tqdm
import csv
import re
import matplotlib.pyplot as plt
import numpy as np


####### init funcs
load_dotenv()

def print_intro_logo():
    logo = """
    LLL      LLL      MMMM   MMMM
    LLL      LLL      MMMMM MMMMM
    LLL      LLL      MMMMMMMMMMM
    LLL      LLL      MMM  M  MMM
    LLLLLLL  LLLLLLL  MMM     MMM
    LLLLLLL  LLLLLLL  MMM     MMM
    """
    print(logo)
    print("Welcome to the LLM Comparison Tool!")
    print(
        "This program compares 2 Large Language Models (LLMs) for performance and accuracy.\n"
    )


def init_redis(host, port):
    return redis.Redis(host=host, port=port, decode_responses=True)


def init_model(gguf):
    return GPT4All(model_name=gguf, model_path=model_path)


####### init vars #######
model_path = "C:\\Users\\omert\\AppData\\Local\\nomic.ai\\GPT4All"
model1_gguf = "mistral-7b-openorca.Q4_0.gguf"
model2_gguf = "mpt-7b-chat-merges-q4_0.gguf"
redis_host = "localhost"
redis_port = 6379

questions_path = "./General_Knowledge_Questions.csv"

# Initialize clients
print_intro_logo()

with tqdm(total=3, desc="Initializing Clients", colour="YELLOW") as pbar:
    redis_client = init_redis(host=redis_host, port=redis_port)
    pbar.update(1)
    model1 = {
        "name": "mistral OpenOrca",
        "model": init_model(model1_gguf),
        "color": color.Fore.LIGHTBLUE_EX,
    }
    pbar.update(1)
    model2 = {
        "name": "MPT Chat",
        "model": init_model(model2_gguf),
        "color": color.Fore.LIGHTCYAN_EX,
    }
    pbar.update(1)

models = [model1, model2]
wolframalpha_Client = wolframalpha.Client(os.environ.get("WOLFRAM_ALPHA_KEY"))


####### helpers #######


def read_csv(path):
    questions = []
    with open(path, "r") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            category, question = row
            questions.append({"Question": question, "Category": category})
    return questions


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # in milliseconds
        return result, elapsed_time

    return wrapper


def create_prompt_from_question(question):
    prompt = use_prompt_tamplate(question)
    return prompt


def use_prompt_tamplate(prompt):
    return f"USER: {prompt}\nASSISTANT: "


def calc_average(value, questions_answerd_num, correctness):
    return (
        value * (questions_answerd_num - 1) + float(correctness)
    ) / questions_answerd_num


def update_lowest_rated(lowestRated, model_name, question, answer, correctness):
    current_lowest_correctness = lowestRated[model_name]["Correctness"]
    if current_lowest_correctness == None:
        swap_lowest_rated_values(
            lowestRated, model_name, question, answer, float(correctness)
        )

    elif float(correctness) < float(current_lowest_correctness):
        swap_lowest_rated_values(
            lowestRated, model_name, question, answer, float(correctness)
        )


def swap_lowest_rated_values(lowestRated, model_name, question, answer, correctness):
    lowestRated[model_name]["Correctness"] = correctness
    lowestRated[model_name]["Question"] = question
    lowestRated[model_name]["Answer"] = answer


def extract_floats(correctness):
    # regex pattern matches a float number.
    pattern = r"\d+\.\d+|\d+"
    matches = re.findall(pattern, correctness)
    float_correctness = " ".join(matches) if matches else ""
    return float_correctness


####### stages #######

@timeit
def get_LLM_Response(model, prompt):
    try:
        response = model["model"].generate(
            prompt,
            max_tokens=50,
            temp=0.7,
            top_k=40,
            top_p=0.4,
            repeat_penalty=1.18,
            repeat_last_n=64,
            n_batch=8,
            n_predict=None,
            streaming=False,
        )
        response = str(response).strip()
        return response
    except:
        raise Exception(f"error with model {model['name']} response")

def get_correct_Answer(question):
    try:
        # Check if the response is in Redis
        cached_response = redis_client.get(question)
        if cached_response:
            return cached_response

        # calling wolfram alpha
        answer = wolframalpha_Client.query(question)
        answer = next(answer.results).text
        # Cache the response for 4 hours (14400 seconds)
        redis_client.setex(question, 14400, answer)

        return answer
    except:
        return None


def get_correctness(question, answer, correct_answer, model=models[0]):
    query = f"Here is a question: {question} . Do not try to solve the question.check similarity of two answers for this questions. Output a number on a scale of 0-1.0 how similar the two answers are. Here are the two answers: 1.{correct_answer} and 2. {answer}."
    prompt = create_prompt_from_question(query)
    try:
        output = model["model"].generate(prompt, max_tokens=3, temp=0.5)
        output = extract_floats(output)
        return output

    except:
        raise Exception(f"error with model {model['name']} response")
    

####### stats #######


def show_stats(
    questions_answerd_num, data_collection, average_correctness, lowestRated
):
    print(color.Fore.GREEN + f"Number of questions answered: {questions_answerd_num}")
    for model in models:
        correctness = average_correctness[model["name"]]
        print(
            model["color"] + f"Average answer rating of {model['name']}: {correctness}"
        )
        lowest_rated_question, lowest_rated_answer, _ = lowestRated[
            model["name"]
        ].values()
        print(
            model["color"]
            + f"Lowest rating question and answer of {model['name']} : {lowest_rated_question} {lowest_rated_answer}"
        )

    cmd = None
    while cmd != "EXIT":
        cmd = input(
            color.Fore.WHITE
            + "CHOOSE STATS: 1 (for Q&A) | 2 (for AVG_CORRECTNESS) | 3 (for CATEGORY_CORRECTNESS)\n4 (for AVG_TIME) | EXIT (for exit)\n"
        )
        match cmd:
            case "1":
                generator_print_q_a(data_collection=data_collection)
            case "2":
                plot_correctness(average_correctness=average_correctness)
            case "3":
                plot_correctness_category(data_collection=data_collection)
            case "4":
                plot_average_time_per_model(data_collection=data_collection)


def plot_average_time_per_model(data_collection):
    model_times = {}

    # Calculate average time for each model across all categories and questions
    for _, questions in data_collection.items():
        for _, answers in questions.items():
            for answer in answers:
                model_name = answer["Model"]
                time_in_ms = answer["TimeInMillisecondsToGetAnswer"]

                if model_name not in model_times:
                    model_times[model_name] = {"total_time": 0, "count": 0}

                model_times[model_name]["total_time"] += time_in_ms
                model_times[model_name]["count"] += 1

    # Prepare data for plotting
    models = list(model_times.keys())
    average_times = [
        model_times[model]["total_time"] / model_times[model]["count"]
        if model_times[model]["count"] > 0
        else 0
        for model in models
    ]

    # Plotting
    _, ax = plt.subplots()
    ax.bar(models, average_times)

    # Add some text for labels, title, etc.
    ax.set_xlabel("Models")
    ax.set_ylabel("Average Time in Milliseconds")
    ax.set_title("Average Response Time per Model")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)

    plt.show()


def plot_correctness_category(data_collection):
    """plot with matplotlib a column graph.
    x axis shows categories, y axis average correctness for this category.
    for each category show two columns showing average correctness values for the 2 models
    """
    categories = list(data_collection.keys())
    model_averages = {}

    # Calculate average correctness for each model in each category
    for category in categories:
        correctness_counts = {}

        for _, answers in data_collection[category].items():
            for answer in answers:
                model_name = answer["Model"]
                correctness = float(answer["Correctness"])

                if model_name not in correctness_counts:
                    correctness_counts[model_name] = {
                        "total_correctness": 0,
                        "count": 0,
                    }

                correctness_counts[model_name]["total_correctness"] += correctness
                correctness_counts[model_name]["count"] += 1

        for model_name, correctness_data in correctness_counts.items():
            average_correctness = (
                correctness_data["total_correctness"] / correctness_data["count"]
                if correctness_data["count"] > 0
                else 0
            )

            if model_name not in model_averages:
                model_averages[model_name] = []

            model_averages[model_name].append(average_correctness)

    # Plotting
    x = np.arange(len(categories))  # the label locations
    width = 0.35 / len(model_averages)  # the width of the bars

    _, ax = plt.subplots()
    for i, (model_name, averages) in enumerate(model_averages.items()):
        ax.bar(x + i * width, averages, width, label=model_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Categories")
    ax.set_ylabel("Average Correctness")
    ax.set_title("Average Correctness by Category and Model")
    ax.set_xticks(x + width * (len(model_averages) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.show()


def plot_correctness(average_correctness):
    
    # Names of the models
    models = list(average_correctness.keys())

    # Correctness values
    correctness_values = list(average_correctness.values())

    
    plt.bar(models, correctness_values)

   
    plt.xlabel("Models")
    plt.ylabel("Correctness")
    plt.title("Correctness of Models")
   
    plt.show()


def data_formatter(data_collection):
    for category in data_collection.keys():
        for question in data_collection[category].keys():
            output = f"question: {question}\n"
            for model in data_collection[category][question]:
                output += f"model {model['Model']} answer: {model['Answer']}\n"
            yield output


def generator_print_q_a(data_collection):
    # Create the generator
    generator = data_formatter(data_collection)
    user_input = ""
    for q_a in generator:
        print(q_a)
        user_input = input("next? press any key | n for exit\n")
        if user_input == "n":
            break


####### main #######


def main():
    questions = read_csv(questions_path)[:5]
    data_collection = (
        {}
    )  # format : {category: {question : [{"Model": model["name"], "Answer": answer, "TimeInMillisecondsToGetAnswer": time_elapsed, "Correctness": correctness},],}}
    lowestRated = {
        model["name"]: {"Question": None, "Answer": None, "Correctness": None}
        for model in models
    }
    questions_answerd_num = 0
    average_correctness = {model["name"]: 0.0 for model in models}

    with tqdm(questions, desc="comparing models", position=0, colour="#068a06") as pbar:
        for question_num, question_object in enumerate(questions):
            with tqdm(
                total=100, desc=f"question {question_num}", leave=False, position=1, colour="GREEN"
            ) as inner_pbar:
                question = question_object["Question"]
                category = question_object["Category"]

                inner_pbar.set_description("Create Prompt")
                prompt = create_prompt_from_question(question)
                inner_pbar.update(5)
                try:
                    inner_pbar.set_description("Models Generating")
                    # asking wolframalpha for correct answer
                    correct_answer = get_correct_Answer(question)
                    inner_pbar.update(15)
                    if correct_answer is None:
                        inner_pbar.set_description("wolframalpha didn't answerd:(")
                        pbar.update()
                        continue
                    questions_answerd_num += 1
                    for model in models:
                        # asking models for answer
                        inner_pbar.set_description(f"{model['name']} Generating")
                        answer, elapsed_time =  get_LLM_Response(model=model, prompt=prompt)
                        inner_pbar.update(15)

                        # correctness
                        inner_pbar.set_description(
                            f"checking correctness for {model['name']}"
                        )
                        correctness = get_correctness(question=question, answer=answer, correct_answer=correct_answer)
                        inner_pbar.update(15)

                        # calculate and update average_correctness
                        inner_pbar.set_description("calculating average correctness")
                        average_correctness[model["name"]] = calc_average(
                            value=average_correctness[model["name"]],
                            questions_answerd_num=questions_answerd_num,
                            correctness=correctness,
                        )
                        inner_pbar.update(2)

                        # update lowest Rated
                        update_lowest_rated(
                            lowestRated=lowestRated,
                            model_name=model["name"],
                            question=question,
                            answer=answer,
                            correctness=correctness,
                        )
                        inner_pbar.update(3)
                        # store data
                        inner_pbar.set_description("store data")
                        answer_data = {
                            "Question": question,
                            "Model": model["name"],
                            "Answer": answer,
                            "TimeInMillisecondsToGetAnswer": elapsed_time,
                            "Correctness": correctness,
                        }
                        if category in data_collection.keys():
                            if question in data_collection[category].keys():
                                data_collection[category][question].append(answer_data)
                            else:
                                data_collection[category][question] = [answer_data]
                        else:
                            data_collection[category] = {question: [answer_data]}
                        inner_pbar.update(5)
                except Exception:
                    print(traceback.format_exc())
            pbar.update()

    show_stats(
        data_collection=data_collection,
        questions_answerd_num=questions_answerd_num,
        average_correctness=average_correctness,
        lowestRated=lowestRated,
    )


if __name__ == "__main__":
    main()
