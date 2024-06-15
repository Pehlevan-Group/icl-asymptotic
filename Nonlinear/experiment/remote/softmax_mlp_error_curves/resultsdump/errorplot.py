import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

mydir = sys.argv[1]
d = int(sys.argv[2])
experimentdata = []
for i in range(66):
    file_path = f'./{mydir}/error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

#experimentdata = np.array(experimentdata)
#print([list(i) for i in experimentdata])
# taus = np.array([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,2,2.5,3,5,7,10])[:14]
# ar40_14 = np.array([[2.6023108959198, 2.254554033279419, 2.3578929901123047, 2.365234613418579, 2.3919875621795654], [2.532776355743408, 2.632019281387329, 2.9205522537231445, 3.0767769813537598, 3.263913631439209], [2.9938747882843018, 3.4875104427337646, 3.4828484058380127, 3.7137277126312256, 3.3197970390319824], [4.53316068649292, 3.8280651569366455, 3.599116563796997, 4.275094985961914, 3.7537589073181152], [5.490716457366943, 5.706916809082031, 5.9682135581970215, 5.557855606079102, 4.82590913772583], [8.540902137756348, 9.850943565368652, 9.8276948928833, 7.452513217926025, 9.802535057067871], [15.348734855651855, 11.570633888244629, 13.046859741210938, 16.848148345947266, 13.666263580322266], [8.493037223815918, 7.10896635055542, 9.352099418640137, 7.154389381408691, 9.944311141967773], [6.644514560699463, 6.023464679718018, 5.859210014343262, 6.839303493499756, 5.901430606842041], [5.004908084869385, 5.914784908294678, 4.98831033706665, 3.6546270847320557, 3.966808795928955], [3.069544553756714, 3.071014881134033, 2.365328073501587, 2.923194169998169, 4.144570827484131], [1.3658015727996826, 1.4078837633132935, 1.3001948595046997, 1.5264066457748413, 1.3517885208129883], [1.0327680110931396, 1.0239278078079224, 1.0031344890594482, 1.018096923828125, 1.0088270902633667], [0.8847051858901978, 0.8775529265403748, 0.8666092157363892, 0.9036922454833984, 0.8679669499397278]])
# ar20_14 = np.array([[2.9062163829803467, 2.2754127979278564, 1.9847791194915771, 2.24310564994812, 2.2531774044036865], [2.9500417709350586, 3.017025947570801, 3.0661983489990234, 2.6269407272338867, 2.3949756622314453], [2.734025001525879, 3.192457675933838, 3.9195914268493652, 2.9441511631011963, 2.9627633094787598], [3.5635297298431396, 2.7771036624908447, 3.7324154376983643, 2.883497953414917, 3.8456647396087646], [5.4376606941223145, 4.761256694793701, 4.455544471740723, 4.692875862121582, 4.328115463256836], [6.895664215087891, 6.2264227867126465, 4.927656650543213, 7.1644110679626465, 5.119468688964844], [8.83810043334961, 6.995192050933838, 7.904427528381348, 8.932870864868164, 8.873292922973633], [11.92676067352295, 11.447528839111328, 17.021238327026367, 10.236629486083984, 11.651045799255371], [9.799042701721191, 10.481451034545898, 10.932585716247559, 7.754958152770996, 9.58895206451416], [7.214998722076416, 6.599148273468018, 8.951101303100586, 6.567244052886963, 5.545844554901123], [4.959431171417236, 5.098135948181152, 4.230360507965088, 4.609433650970459, 6.309449672698975], [1.8140617609024048, 2.240433692932129, 3.1813955307006836, 2.4810447692871094, 2.3843579292297363], [1.932372808456421, 1.1624127626419067, 2.258294105529785, 1.2005141973495483, 1.3038829565048218], [0.9982557892799377, 1.029316782951355, 1.0299745798110962, 1.4841887950897217, 1.9276036024093628]])

taus = np.array(list(np.linspace(0.5,6.5,61)) + [7,8.5,9,9.5,10]) #np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.75,2,2.25,2.5,3,3.5,4,4.5,5])
#np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,2,2.5,3,5]);

tau40 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.75, 2.0, 2.25, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 6.0, 7.0, 8.0, 10.0]
mean40 = np.array([1.8644035816192628, 1.9722130298614502, 2.035210919380188, 2.101726603507996, 2.205843210220337, 2.2247981071472167, 2.330627918243408, 2.323292112350464, 2.4346358299255373, 2.420562219619751, 2.479092168807983, 2.490829849243164, 2.5157453060150146, 2.6267337799072266, 2.7364365577697756, 2.604920530319214, 2.668213129043579, 2.738864374160767, 2.67364239692688, 2.7134376525878907, 2.8647740364074705, 2.855447292327881, 3.076593255996704, 3.1373746395111084, 3.0109063625335692, 3.22276291847229, 3.244851541519165, 3.1987114429473875, 3.286079692840576, 3.2807680130004884, 3.2416200637817383, 3.109679698944092, 3.037140989303589, 3.021123266220093, 3.0342694759368896, 3.0439451694488526, 2.814418649673462, 2.8608307361602785, 2.4403477191925047, 2.059476351737976, 1.7929187774658204, 1.5477179527282714])
std40 = np.array([0.046938467210376825, 0.057345878825491366, 0.09831141851813693, 0.08569056534983971, 0.07333785173845316, 0.08196416797839998, 0.0882559490492408, 0.1078536152998083, 0.049290140714586116, 0.061565208639441225, 0.11169014917559021, 0.089381928001992, 0.03971062521754454, 0.10941578276181098, 0.11068817514580398, 0.06863722980287401, 0.08040381830021674, 0.11746392410882953, 0.1091447695996017, 0.073544981534382, 0.20058096853173132, 0.17724683605974512, 0.04517649955206504, 0.13273980377593209, 0.21244002929896785, 0.1680499006358194, 0.09062710572997384, 0.20210277788050227, 0.07892524391775459, 0.058290737126364275, 0.15535654991701817, 0.09705839927514807, 0.06155429419156726, 0.07146260248477311, 0.31767985881452093, 0.2359274542388308, 0.1485443549658341, 0.13580788386101167, 0.14194100980818686, 0.1542629861226431, 0.08231140657099544, 0.10231828442321068])
means = np.array([np.mean(experimentdata[i]) for i in range(len(experimentdata))])
stds = np.array([np.std(experimentdata[i]) for i in range(len(experimentdata))])

## PLOTS !!!!!

# DEFINE STANDARD FORMATING FOR FIGURES USED THROUGHOUT PAPER
sns.set(style="white",font_scale=2.5,palette="colorblind")
plt.rcParams['lines.linewidth'] = 4
plt.rcParams["figure.figsize"] = (12, 10)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
myred = '#8B0000'
colors = [myred, color_cycle[1], color_cycle[2]]
plt.gca().spines['top'].set_color('lightgray')
plt.gca().spines['right'].set_color('lightgray')
plt.gca().spines['bottom'].set_color('lightgray')
plt.gca().spines['left'].set_color('lightgray')

plt.plot(taus,means,label=f'd = {d}',color=myred)
plt.fill_between(taus, means-stds, means+stds, color=myred, alpha = 0.2)
plt.plot(tau40,mean40,label=f'd = 40',color=color_cycle[1])
plt.fill_between(tau40, mean40-std40, mean40+std40, alpha = 0.2,color=color_cycle[1])

# Nice legend
leg = plt.legend()
leg.get_frame().set_alpha(0)
# Axis Formatting
plt.xlabel(r'$\tau = n/d^2$')
plt.ylabel(r'$e^{ICL}(\Gamma^*)$')
plt.xticks(fontsize=20);
plt.yticks(fontsize=20);
plt.savefig("icl_dd_softmlp_x2.pdf", bbox_inches='tight')

