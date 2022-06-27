from tkinter import *
import json

screen = Tk()
screen.geometry("400x200")
screen.title("Data Collection")

def commit_changes(intents):
    with open("Data/intents.json", "w") as f:
        json.dump(intents, f, indent=4)

def add_data(tag, question, answer):
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    for tags in intents["intents"]:
        if tag == tags["tag"]:
            if question not in tags["patterns"] and len(question) >= 1:
                tags["patterns"].append(question)
            if answer not in tags["responses"] and len(answer) >= 1:
                tags["responses"].append(answer)
            commit_changes(intents)
            return

    intents["intents"].append({"tag": tag, "pattern":[question], "responses":[answer]})
    commit_changes(intents)
    

tag = Entry(screen, borderwidth=0, justify="center")
tag.pack(pady=20)

question = Entry(screen, borderwidth=0, justify="center")
question.pack(pady=10)

answer = Entry(screen, borderwidth=0, justify="center")
answer.pack(pady=10)

submit = Button(screen, borderwidth=1, text="Add data", command=lambda:add_data(tag.get(), question.get(), answer.get()))
submit.pack(pady=10)

screen.mainloop()