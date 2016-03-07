from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import *
from Tkinter import *
import cart.cart as cart
import matplotlib
matplotlib.use('TkAgg')

def reDraw(tolS, tolN, lam):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        modelTree = cart.createTree(reDraw.trainDataMatrix, lam, cart.modelLeaf, cart.modelErr, (tolS, tolN))
        if chkBtnPostpruneVar.get():
            modelTree = cart.prune(modelTree, reDraw.testDataMatrix, cart.modelTreeEval,lam)
        yHat = cart.createForecast(modelTree, reDraw.printDataMatrixXcoord, cart.modelTreeEval)
        print "model tree(tolN:{},tolS:{},postprune:{}) is: [Test RSS:{}\tTrain RSS:{}]".format(tolN,
                                                                                                tolS,
                                                                                                chkBtnPostpruneVar.get(),
                                                                                                cart.calcRss(modelTree, reDraw.testDataMatrix, cart.modelTreeEval),
                                                                                                cart.calcRss(modelTree, reDraw.trainDataMatrix, cart.modelTreeEval))

    else:
        regressionTree = cart.createTree(reDraw.trainDataMatrix, lam, cart.regLeaf, cart.regErr,(tolS, tolN))
        if chkBtnPostpruneVar.get():
            regressionTree = cart.prune(regressionTree, reDraw.testDataMatrix, cart.regTreeEval,lam)
        yHat = cart.createForecast(regressionTree, reDraw.printDataMatrixXcoord, cart.regTreeEval)
        print "regression tree(tolN:{},tolS:{},postprune:{}) is: [Test RSS:{}\tTrain RSS:{}]".format(tolN,
                                                                                                     tolS,
                                                                                                     chkBtnPostpruneVar.get(),
                                                                                                     cart.calcRss(regressionTree, reDraw.testDataMatrix, cart.regTreeEval),
                                                                                                     cart.calcRss(regressionTree, reDraw.trainDataMatrix, cart.regTreeEval))
    reDraw.a.scatter(reDraw.trainDataMatrix[:,0], reDraw.trainDataMatrix[:,1], s=5)
    reDraw.a.plot(reDraw.printDataMatrixXcoord, yHat, linewidth=2.0)
    reDraw.canvas.show()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print "enter Integer for tolN"
        tolNentry.delete(0, END)
        tolNentry.insert(0,'10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print "enter Float for tolS"
        tolSentry.delete(0, END)
        tolSentry.insert(0,'1.0')
    return tolN, tolS


def getLambda():
    try:
        lam = float(lambdaEntry.get())
    except:
        lam = 1.0
        print "enter Float for tolN"
        lambdaEntry.delete(0, END)
        lambdaEntry.insert(0,'1.0')
    return lam

def drawNewTree():
    tolN, tolS = getInputs()
    lam = getLambda()
    reDraw(tolS, tolN, lam)



if __name__ == '__main__':
    root = Tk()

    reDraw.f = Figure(figsize=(5,4), dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.show()
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

    Label(root, text="tolN").grid(row=1, column=0)
    tolNentry = Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0,'10')
    Label(root, text="tolS").grid(row=2, column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0,'1.0')
    Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2,rowspan=3)

    chkBtnVar = IntVar()
    chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)

    Label(root, text="lambda").grid(row=4, column=0)
    lambdaEntry = Entry(root)
    lambdaEntry.grid(row=4, column=1)
    lambdaEntry.insert(0,'1.0')

    Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2,rowspan=3)


    chkBtnPostpruneVar = IntVar()
    chkBtnPostprune = Checkbutton(root, text="Post Prune", variable = chkBtnPostpruneVar)
    chkBtnPostprune.grid(row=3, column=1, columnspan=2)

    reDraw.trainDataMatrix = mat(cart.loadData('./data/bikeSpeedVsIq_train.txt'))
    reDraw.testDataMatrix = mat(cart.loadData('./data/bikeSpeedVsIq_test.txt'))
    reDraw.printDataMatrixXcoord = arange(min(reDraw.trainDataMatrix[:,0]), max(reDraw.trainDataMatrix[:,0]),0.01)
    reDraw(1.0, 10, 1.0)

    Button(root, text='Quit',fg="black", command=root.quit).grid(row=1, column=2)
    root.mainloop()