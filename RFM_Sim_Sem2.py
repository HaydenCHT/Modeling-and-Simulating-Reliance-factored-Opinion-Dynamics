import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
#from google.oauth2 import service_account
#from googleapiclient.discovery import build
#from gspread_dataframe import set_with_dataframe
from decimal import Decimal
#from scipy import stats
import multiprocessing

from functions import update_reliance
#from functions import linear_shift_after_converge
from functions import export_to_sheets
from functions import update_opinions_RFM
from functions import linear_shift_before_converge
from functions import insert_agent
from functions import read_real_network



def plot_for_reference(all_op, num, epsilon, model, network, adj):
    plt.figure(figsize=(10, 6))
    for i in range(num):
        plt.plot(all_op[:, i])

    plt.xlabel('Iterations')
    plt.ylabel('Opinion')
    plt.title(f'Opinion Evolution in the {model}' , fontweight='bold', pad=36)
    plt.figtext(0.5, 0.9,
                f'Number of individuals: {num}          Epsilon: {epsilon}           Network: {network}           Adjustment Factor: {adj}',
                ha='center', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.5'))
    plt.grid(True)
    plt.show()


def plot_all_testcase_opinions(all_op, num, epsilon, model, network, early_ts, manipulate_to):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    for i in range(len(all_op)):
        arr = np.array(all_op[i][1])
        for j in range(num):
            axs[i].plot(arr[:, j])
        axs[i].set_xlabel('Iterations')
        axs[i].set_ylabel('Opinion')
        axs[i].set_title("Manipulation starting early at " + all_op[i][0])
        axs[i].grid(True)
    #plt.title(f'e: {epsilon}; change: {shift_rate}; adj: {adj}')
    plt.tight_layout()
    plt.savefig(f'Real-Network-Graphs.png')
    plt.close()


def plot_reliance_for_reference(all_rel, num, epsilon, model, network, adj):
    plt.figure(figsize=(10, 6))
    for i in range(num):
        plt.plot(all_rel[:, i, target_subject-1])

    plt.xlabel('Iterations')
    plt.ylabel('Reliance')
    plt.title(f'Population\'s Reliance Evolution on Individual {target_subject} in the {model}' , fontweight='bold', pad=36)
    plt.figtext(0.5, 0.9,
                f'Number of individuals: {num}          Epsilon: {epsilon}           Network: {network}           Adjustment Factor: {adj}',
                ha='center', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.5'))
    plt.grid(True)
    plt.show()



#-------------------------------------------------------------------------------

def process_test_case(test_case, 
                      opinions_set, 
                      R_set, network_set, num_individuals, num_opinions, 
                      num_rand_structures, 
                      #num_iterations, 
                      epsilon, round_op_to_decimal, convergence_boundary, dbscan, misinfo_agent):
    
    adj_factor = test_case["adj_factor"]
    case = test_case["case"]
    #alpha = test_case["alpha"]
    #beta = test_case["beta"]
    early_t = test_case['early_t']
    MA_0_op = test_case['MA_0_op']
    num_iterations = test_case['num_iter']
    shift_rate = float((Decimal('1') - Decimal(MA_0_op))/Decimal(str(num_iterations - early_t)))
    

    num_cluster_formations = []
    all_initial_sum_opinion = []
    stabilize_timestamps_phase_1 = []
    t1_op = []
    t1_conn = []
    t1_trust_conn = []
    stabilize_timestamps_phase_2 = []
    time_diff = []
    stabled_cluster_sizes = []
    stabled_cluster_sizes_2 = []
    fail_manipulate = []
    runtime_err = []
    all_op_var = []

    for i in range(num_opinions):
        opinions = np.copy(opinions_set[i])
        
        #opinions = np.random.beta(a=alpha, b=beta, size=(num_individuals))
        op_var = np.var(opinions)
        initial_op_sum = np.sum(opinions)
        #t = t1 * initial_op_sum + t2
        #shift_rate = t3 / t

        sum_clusters = 0
        sum_stabled_cluster_size = 0
        sum_stabled_cluster_size_2 = 0
        sum_fail = 0
        sum_converge_time_phase_1 = 0
        sum_t1_op = 0
        sum_t1_conn = 0
        sum_t1_trust_conn = 0
        sum_converge_time_phase_2 = 0
        t_diff = []
        sum_rt_err = 0

        for r in range(num_rand_structures):
            #misinfo_agent = -1
            A = np.copy(network_set[r])
            column_A_sums = np.sum(A, axis=0)
            if case == 1:
                misinfo_agent = np.argmax(column_A_sums) + 1
            else:
                misinfo_agent = np.argmin(column_A_sums) + 1
            insert_agent(A=A, agent_index=misinfo_agent-1)
            opinions[misinfo_agent-1] = float(MA_0_op)
            R = np.copy(R_set[0]) * A
            #if full_self_reliance:
             #   np.fill_diagonal(R, 1)


            converge_time_phase_1 = 0
            converge_time_phase_2 = 0
            curr_opinions = np.copy(opinions)
            t1_all_op = curr_opinions
            all_opinions = [curr_opinions]
            curr_R = np.copy(R)
            t1_all_rel = curr_R
            #all_reliance = [curr_R]

            for h in range(1, num_iterations + 1):
                new_opinions, converge_time_phase_1, converge_time_phase_2, A, misinfo_agent = linear_shift_before_converge(
                    early_t, shift_rate, case, A, curr_R, curr_opinions, epsilon, h, converge_time_phase_1, converge_time_phase_2, manipulate_to, misinfo_agent, round_op_to_decimal
                )
                #new_opinions, converge_time_phase_1, converge_time_phase_2, A, misinfo_agent = linear_shift_after_converge(
                 #   case, A, curr_R, curr_opinions, epsilon, h, converge_time_phase_1, converge_time_phase_2, manipulate_to, misinfo_agent, shift_rate, round_op_to_decimal
                #)
                #new_opinions = update_opinions_RFM(A, curr_R, curr_opinions, epsilon)
                #if not np.array_equal(np.sort(np.round(new_opinions, round_op_to_decimal)), np.sort(np.round(curr_opinions, round_op_to_decimal))):
                 #   converge_time_phase_1 = h + 1

                all_opinions.append(new_opinions)
                curr_opinions = new_opinions
                new_R = update_reliance(A, curr_R, curr_opinions, epsilon, adj_factor)
                #all_reliance.append(new_R)
                curr_R = new_R
                if h == converge_time_phase_1:
                    t1_all_op = curr_opinions
                    t1_all_rel = curr_R
            
            labels = dbscan.fit_predict(curr_opinions.reshape(-1, 1))
            sum_clusters += len(set(labels) - {-1})
            #t1_cluster = np.count_nonzero(np.abs(all_opinions[converge_time_phase_1 - 1] - all_opinions[converge_time_phase_1 - 1][misinfo_agent - 1]) <= 0.005)
            t1_cluster = np.count_nonzero(np.abs(t1_all_op - t1_all_op[misinfo_agent - 1]) <= 0.005)
            sum_stabled_cluster_size += t1_cluster
            sum_t1_op += t1_all_op[misinfo_agent - 1]
            sum_t1_conn += np.sum(A[:, misinfo_agent - 1])
            sum_t1_trust_conn += np.sum(t1_all_rel[:, misinfo_agent - 1])
            if converge_time_phase_2 > 0:
                t2_cluster = np.count_nonzero((curr_opinions >= 0.95) & (curr_opinions <= 1))
                sum_stabled_cluster_size_2 += t2_cluster
                if t1_cluster != 1 and t1_cluster > t2_cluster:
                    sum_fail += 1


            sum_converge_time_phase_1 += converge_time_phase_1
            sum_converge_time_phase_2 += converge_time_phase_2
            if converge_time_phase_1 != 0 and converge_time_phase_2 != 0:
                t_diff.append(converge_time_phase_2 - converge_time_phase_1)
            if converge_time_phase_1 == num_iterations + 1:
                sum_rt_err += 1
            if i == 0 and r == 0:
                testcase_opinions = [str(early_t), all_opinions]

        num_cluster_formations.append(sum_clusters / num_rand_structures)
        all_initial_sum_opinion.append(initial_op_sum)
        stabilize_timestamps_phase_1.append(sum_converge_time_phase_1 / num_rand_structures)
        t1_op.append(sum_t1_op / num_rand_structures)
        t1_conn.append(sum_t1_conn / num_rand_structures)
        t1_trust_conn.append(sum_t1_trust_conn / num_rand_structures)
        stabilize_timestamps_phase_2.append(sum_converge_time_phase_2 / num_rand_structures)
        time_diff.append(' '.join(map(str, t_diff)) if t_diff else '')
        stabled_cluster_sizes.append(sum_stabled_cluster_size / num_rand_structures)
        stabled_cluster_sizes_2.append(sum_stabled_cluster_size_2 / num_rand_structures)
        fail_manipulate.append(sum_fail)
        runtime_err.append(sum_rt_err)
        all_op_var.append(op_var)

    data = {
        'Number of Individuals': [num_individuals] * num_opinions,
        'Number of Iterations': [num_iterations] * num_opinions,
        'Confidence Interval': [epsilon] * num_opinions,
        'Convergence': [convergence_boundary] * num_opinions,
        'Opinion Set': [i + 1 for i in range(num_opinions)],
        'Model': ["RFM"] * num_opinions,
        'Adjustment Factor': [adj_factor] * num_opinions,
        'Number of Clusters': num_cluster_formations,
        'Initial Sum of opinion': all_initial_sum_opinion,
        'Convergence Time Phase 1': stabilize_timestamps_phase_1,
        'Op at t1': t1_op,
        'Conn at t1': t1_conn,
        'Trust Conn at t1': t1_trust_conn,
        'Convergence Time Phase 2': stabilize_timestamps_phase_2,
        'Conv Time Diff': time_diff,
        'Size of opinion cluster of MA c1': stabled_cluster_sizes,
        'Size of opinion cluster of MA c_max': stabled_cluster_sizes_2,
        'Fail Mani': fail_manipulate,
        'Runtime Errors': runtime_err,
        'case': [case] * num_opinions,
        #'alpha': [alpha] * num_opinions,
        #"beta" : [beta] * num_opinions,
        'var': all_op_var,
        'shift_rate': [shift_rate] * num_opinions,
        'MA op at t0': [MA_0_op] * num_opinions,
        'op type': 'uni'
    }
    df = pd.DataFrame(data)
    # print(f'output case {case}')
    return df, testcase_opinions

# Main code
num_individuals = 100
num_opinions = 1
num_rand_structures = 1
# num_iterations = 600
epsilon = 0.1
target_subject = 1
manipulate_to = 1
round_op_to_decimal = 3
convergence_boundary = 1 * (10 ** (-round_op_to_decimal))
dbscan = DBSCAN(eps=convergence_boundary, min_samples=2)
misinfo_agent = -1

test_cases = [
    #{"case": 1, "adj_factor": 0.1, "full_self_reliance": False, "t1": 66.1825767, "t2": 1170.335997, "t3": 0.5},
    
    {"case": 1, "adj_factor": 0.1, "MA_0_op": "0.5", "early_t": 0, "num_iter": 2000},
    {"case": 1, "adj_factor": 0.1, "MA_0_op": "0.5", "early_t": 80, "num_iter": 2000},

    {"case": 2, "adj_factor": 0.1, "MA_0_op": "0.5", "early_t": 0, "num_iter": 2000},
    {"case": 2, "adj_factor": 0.1, "MA_0_op": "0.5", "early_t": 80, "num_iter": 2000},



    #{"case": 1, "adj_factor": 0.1, "full_self_reliance": False, "t1": 6.115473501, "t2": 114.2642746, "t3": 0.1/1.5},
]

#opinions_set = np.random.rand(num_opinions, num_individuals)
#R_set = np.random.rand(num_opinions, num_individuals, num_individuals)
#network_set = [np.ones((num_individuals, num_individuals))]
network_set, R_set, num_individuals = read_real_network('Last Last E/moreno_seventh/out.moreno_seventh_seventh')

#for networks in network_set:
 #   for network in networks:
  #      np.fill_diagonal(network, 1)

G = nx.DiGraph()
G.add_nodes_from(range(num_individuals))

for i in range(num_individuals):
    for j in range(num_individuals):
        if network_set[0][i][j] == 1:
            G.add_edge(i, j, weight=R_set[0][i][j])
pos = nx.kamada_kawai_layout(G)
node_colors = ['orange' if node == misinfo_agent-1 else 'lightblue' for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
column_sums = np.sum(network_set[0], axis=0)
for i in range(2):
    if i:
        misinfo_agent = np.argmin(column_sums) + 1
    else:
        misinfo_agent = np.argmax(column_sums) + 1
    
    inward_edges = [(u, v) for u, v in G.edges() if v == misinfo_agent-1]

    edge_colors = ['red' if edge in inward_edges else 'grey' for edge in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=10)

    plt.title(f'Moreno Seventh Graders')
    plt.axis('off')
    plt.savefig(f'network_diagram_{i}')

opinions_set = np.random.rand(num_opinions, num_individuals)#np.random.beta(a=4, b=6, size=(num_opinions, num_individuals))#np.random.rand(num_opinions, num_individuals)#

all_testcase_opinions = []

if __name__ == "__main__":
    args_list = [
        (test_case, 
         opinions_set,
         R_set, network_set, num_individuals, num_opinions, 
         num_rand_structures, 
         #num_iterations, 
         epsilon, round_op_to_decimal, 
         convergence_boundary, dbscan, misinfo_agent) 
        for test_case in test_cases
    ]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_test_case, args_list)

    combined_df = pd.concat([result[0] for result in results], ignore_index=True)
    all_testcase_opinions = [result[1] for result in results]
    #export_to_sheets(combined_df, "E3.4 7grade")

    early_ts = [e['early_t'] for e in test_cases]
    plot_all_testcase_opinions(all_testcase_opinions, num_individuals, epsilon, "RFM", "Real", early_ts, manipulate_to)