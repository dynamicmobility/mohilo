I want you to create a ipad class/function which spawns a survey for the user to take. Here are the requirements

1. The survey line should go from 1-5, with a sad face on 1 and a happy face on 5
2. When the user's response is not required, the interface should grey out
3. When the user's response is required, the interface should become interactable. Moreover, a timer should start for 25 seconds. 
4. the user is required to select a position on the slider and hit a 'submit' button before this timer runs out. if they do not do this, no value is recorded from that particular session.
5. This function should be wired into logger_example.py, where the ipad survery on comfort function currently lives. it is currently commented out. For now, simply comment out the metabolic stuff and only do the comfort stuff (just for testing and proof of concept).
6. the interface should look similar-ish to what is currently in tablet/. 
7. this will all need to work in a worker thread (see logger).

Let me know what questions you have
