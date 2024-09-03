import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

mydir = sys.argv[1]
d = int(sys.argv[2])
icl = []
inds = range(32) #[a for a in range(13)] + [a for a in range(25,34)]
for i in inds:
    file_path = f'./{mydir}/icl-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    icl.append(numbers)
# idg = []
# for i in inds:
#     file_path = f'./{mydir}/idg-{i}.txt'
#     # Read the numbers from the file and convert them to floats
#     with open(file_path, 'r') as file:
#         numbers = [float(line.strip()) for line in file if line.strip()]
#     idg.append(numbers)

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
means_icl = np.array([np.mean(icl[i]) for i in range(len(icl))]);
stds_icl = np.array([np.std(icl[i]) for i in range(len(icl))]);
# means_idg = np.array([np.mean(idg[i]) for i in range(len(idg))]);
# stds_idg = np.array([np.std(idg[i]) for i in range(len(idg))]);
print(list(means_icl))
# print(list(means_idg))
print(list(stds_icl))
# print(list(stds_idg))
# Ks20 = list(range(2,d+1,2)) + list(np.int64(np.log(np.logspace(1.5*d,10*d,30)))); 
# Ks40 = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(10*d),30)));
# Kappasfix = [23, 28, 35, 45, 57];
Ks20 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 28, 35, 45, 57, 69, 82, 96, 109, 123, 136, 150, 163, 177, 190, 204, 217, 231, 244, 258, 271, 285, 298, 312, 325, 339, 352, 366, 379, 393, 406, 420, 433, 447, 460];
Ks20soft = list(range(2,d+5,1)) + list(range(d+6,2*d,2))
#Ks40 = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(100*d),20))) 
Ks40 = list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(5*d),20))) + list([10*d, 50*d])
Ks40soft = np.array(list(range(2,d+22,2)))
Ks80 = np.array(list(range(2,d+1,4)) + list(np.int64(np.logspace(np.log10(d),np.log10(5*d),15))))
Ks80soft = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 44, 52, 60, 70, 80, 90, 100, 120, 140, 160, 200])
if d == 80:
    Ks = np.array(Ks80soft[inds])
if d == 40:
    Ks = np.array(Ks40)[inds]
if d == 20:
    Ks = np.array(Ks20)[inds]
print([k for k in Ks])
kappas = Ks/d;


d=100
dmmse_data = [1.9266200302560967, 1.8945463437174872, 1.906960194062516, 1.8262693894130426, 1.8422270916845187, 1.833511194086416, 1.74618300174723, 1.7467231097130242, 1.7017476767851412, 1.697519553899994, 1.69594987768819, 1.715893616177506, 1.6700208197732995, 1.658500475354113, 1.639018908122997, 1.618160576954193, 1.6094190497207814, 1.5981250975990435]
ridge_data = [0.5181161341193182, 0.5188057133424524, 0.5194021423230276, 0.5209378134567436, 0.5196384784939407, 0.5231398959686532, 0.5178145099345771, 0.5194335203762636, 0.5196678327291469, 0.5178368526622865, 0.5200427672864073, 0.5219742689397068, 0.5185716862314423, 0.5178652492493762, 0.5195682354600782, 0.5167523684423125, 0.5214838533720981, 0.5194597730139145]
K_array = list(np.int64(np.logspace(np.log10(0.01*d),np.log10(10*d),20))); #list(np.int64(np.logspace(np.log10(0.05*d),np.log10(500*d),40))); 
K_array = [i for n, i in enumerate(K_array) if i not in K_array[:n]]
K_array = np.array(K_array)
kappa_ary_bayes = K_array/d;

start = 0

m40 = np.array([1.7812299489974976, 2.1481076743867664, 2.2374595642089843, 2.198765856879098, 2.2810339662763806, 2.4408169269561766, 2.2726720571517944, 2.317395180463791, 2.3158360719680786, 2.3253814458847044, 2.3654044999016657, 2.3375391960144043, 2.2779670476913454, 2.302447954813639, 2.3896350264549255, 2.403994185583932, 2.309493409262763, 2.3179553508758546, 2.3161668181419373, 2.3063486417134604, 2.316216540336609, 2.4062320232391357, 2.3264283180236816, 2.2732573986053466, 2.33262243270874, 2.2810477733612062, 2.319455146789551, 2.345801663398743, 2.3088725328445436, 2.3218998432159426, 2.346046209335327, 2.338757061958313])
#[0.27754528969526293, 1.7077156967586942, 1.970266628265381, 1.9912167617252894, 2.149873389138116, 2.3474225759506226, 2.1690245469411216, 2.2467653453350067, 2.230840712785721, 2.2378470659255982, 2.299055020014445, 2.289285898208618, 2.1933747053146364, 2.2626700666215687, 2.3475275933742523, 2.3770603452410017, 2.277575625313653, 2.2741918325424195, 2.2806462943553925, 2.2925806840260825, 2.2807183504104613, 2.3842634916305543, 2.2969191908836364, 2.2490411758422852, 2.3154441356658935, 2.2620445013046266, 2.2971100568771363, 2.317306399345398, 2.2875981092453004, 2.2981045246124268, 2.3379491567611694, 2.3405978202819826]
s40 = np.array([0.1967613761847934, 0.09628164058881102, 0.1228780689174422, 0.08642270483438062, 0.08086212612704081, 0.1640852705705161, 0.10140412722051097, 0.10651711187275562, 0.08864831234818202, 0.10565972087631381, 0.0915765728011634, 0.13375786754731372, 0.08790368699316928, 0.1599386429131236, 0.07770036377063584, 0.13193448766407956, 0.09647413724368803, 0.07944935788391742, 0.09937578714764571, 0.0857759362655868, 0.10026196891762454, 0.08045921132356146, 0.12012860450539034, 0.085407427378051, 0.1352282645795387, 0.0852349061455536, 0.09982790096325221, 0.1067977267347156, 0.05356368205495572, 0.0824494015844258, 0.09342395420557666, 0.09462949117377202])
#[0.08896747409643699, 0.17160876818391774, 0.18782632282857847, 0.12679163452292294, 0.1250454301287731, 0.1979658148640085, 0.12025900310776197, 0.1278325430736067, 0.11712469233186566, 0.13065975801212926, 0.08862239024781951, 0.16498385751398395, 0.09774141987108438, 0.18093027082417554, 0.1107482383268486, 0.1528623790214789, 0.12041549193427173, 0.0837907180810394, 0.11227161325512222, 0.09611814846636899, 0.12448700725627822, 0.09067981489432547, 0.12666502738221155, 0.09794097954739632, 0.12604223555567978, 0.09608593128321745, 0.09825145899018684, 0.11068904466390858, 0.05297743615271092, 0.09180548315759181, 0.08771262400965497, 0.09444718007345328]
k40 = np.array([2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 40, 43, 47, 51, 56, 61, 66, 72, 78, 85, 93, 101, 110, 120, 130, 142, 155, 168, 183, 200, 400, 2000])/40
plt.scatter(kappas[start:], means_icl[start:],s=100,color = 'orange',label=f'{mydir}')
plt.fill_between(kappas[start:], means_icl[start:]-stds_icl[start:], means_icl[start:]+stds_icl[start:],color='orange',alpha=0.2)
# plt.scatter(k40[start:], m40[start:],s=100,color = 'blue',label='icl 40')
# plt.fill_between(k40[start:], m40[start:]-s40[start:], m40[start:]+s40[start:],color='blue',alpha=0.2)
plt.scatter(kappa_ary_bayes[4:],dmmse_data[4:],s=100,color='red',label='dmmse')
plt.scatter(kappa_ary_bayes[4:],ridge_data[4:],s=100,color='green',label='ridge')
# plt.scatter(kappas[start:], means_idg[start:],color='green',label='idg')
# plt.fill_between(kappas[start:], means_idg[start:]-stds_idg[start:], means_idg[start:]+stds_idg[start:],color='green',alpha=0.2)
plt.axvline(1,linestyle=':',color='grey')
plt.legend()
plt.xscale('log')

# Nice legend
leg = plt.legend()
leg.get_frame().set_alpha(0)
# Axis Formatting
plt.xlabel(r'$\kappa = k/d$')
plt.ylabel(r'$e^{ICL}(\Gamma^*)$')
plt.xticks(fontsize=20);
plt.yticks(fontsize=20);
plt.savefig(f'{mydir}/plot.png', bbox_inches='tight')