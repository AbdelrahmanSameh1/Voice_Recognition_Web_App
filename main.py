from flask import Flask
from flask import request
from flask import render_template, redirect, url_for
from functions import *


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


amrgmm = pickle.load(open("Amr.gmm", "rb"))
besharagmm = pickle.load(open("Beshara.gmm", "rb"))
samehgmm = pickle.load(open("Sameh.gmm", "rb"))

openDoorgmm = pickle.load(open("openTheDoor.gmm", "rb"))
closeDoorgmm = pickle.load(open("closeTheDoor.gmm", "rb"))
opwindowgmm = pickle.load(open("openTheWindow.gmm", "rb"))
clwindowgmm = pickle.load(open("closeTheWindow.gmm", "rb"))


@app.route("/preProcessing")
def preProcessing():

    features = extract_features('output.wav')

    amr = np.array(amrgmm.score(features))
    beshara = np.array(besharagmm.score(features))
    sameh = np.array(samehgmm.score(features))
    opendoor = np.array(openDoorgmm.score(features))
    closedoor = np.array(closeDoorgmm.score(features))
    openwindow = np.array(opwindowgmm.score(features))
    closewindow = np.array(clwindowgmm.score(features))

    wordscore = [opendoor, closedoor, openwindow, closewindow]
    wordresult = np.max(wordscore)

    speakerscore = [amr, beshara, sameh]
    result = np.max(speakerscore)
    # spectral_Rolloff('output.wav', 'spec_Rolloff')
    print(speakerscore)
    plot_barChart(speakerscore, True, ['Amr', 'Beshara', 'Sameh'], 'Speaker')
    plot_barChart(wordscore, False, [
                  'Open the door', 'Close the door', 'Open the window', 'Close the window'], 'Word')
    spectral_features('output.wav', 'spec_features')
    chroma('output.wav')
    zeros('output.wav')

    otherFlag = False
    otherscore = speakerscore - result

    for i in range(len(otherscore)):
        if otherscore[i] == 0:
            continue
        if otherscore[i] > -1:
            otherFlag = True
    if otherFlag == False:
        if result == amr:
            speaker = 'Amr'
        elif result == beshara:
            speaker = "beshara"
        else:
            speaker = "sameh"

    else:
        speaker = 'other'

    if wordresult == opendoor:
        wordIs = "Open"

    else:
        wordIs = "other"

    prediction = speaker + " " + wordIs
    if speaker == 'Amr':
        rolloff = spectral_Rolloff('AudioData/Amr/Amr_openTheDoor (10).wav',
                                   'spec_Rolloff', speaker)
    elif speaker == 'beshara':
        rolloff = spectral_Rolloff(
            'AudioData/Beshara/Beshara_openTheDoor (20).wav', 'spec_Rolloff', speaker)
    elif speaker == 'sameh':
        rolloff = spectral_Rolloff(
            'AudioData/Sameh/Sameh_openTheDoor (1).wav', 'spec_Rolloff', speaker)
    elif speaker == 'other':
        rolloff = spectral_Rolloff(
            '1.wav', 'spec_Rolloff', speaker)

    bars(amr, beshara, sameh)
    pie(amr, beshara, sameh, flag=True)
    if speaker == "other" or wordIs == "other":
        pie(amr, beshara, sameh, flag=False)

    rms('output.wav', 'AudioData/Amr/Amr_openTheDoor (20).wav',
        'AudioData/Beshara/Beshara_openTheDoor (2).wav', 'AudioData/Sameh/Sameh_openTheDoor (30).wav')
    guass(rolloff)
    print(prediction)
    img, fig = plot_melspectrogram('output.wav')
    fig.colorbar(img, format="%+2.f")
    spectro = plt.savefig('./static/spectro.png')
    spectro = True

    return render_template('index.html', prediction="{}".format(prediction), spectro=spectro)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    return redirect(url_for('preProcessing'))


@app.route("/", methods=['GET', 'POST'])
def audio():
    if request.method == "POST":
        fs = 22050  # record sample rate
        seconds = 2  # record duration
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # wait the record to be finished
        write('output.wav', fs, myrecording)
    return redirect(url_for('predict'))


if __name__ == "__main__":
    app.run(debug=True)
