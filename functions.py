import numpy as np
import pandas as pd
from scipy import stats
from google.oauth2 import service_account
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = service_account.Credentials.from_service_account_file("XXX.json", scopes=SCOPES)
service = build("sheets", "v4", credentials=creds)

SPREADSHEET_ID = "XXX"


def export_to_sheets(df, sheet):
        sheet_service = service.spreadsheets()
        body = {
            'values': [df.columns.tolist()] + df.values.tolist()
        }
        sheet_service.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=sheet,  
            valueInputOption="RAW",
            body=body
        ).execute()


def update_opinions_BCM(A, R, op, eps):
    differences = np.abs(op[:, None] - op[None, :])  # Compute pairwise differences
    within_eps = (differences <= eps) & A.astype(bool)  # Apply epsilon threshold and adjacency
    influential = within_eps * op  # Mask opinions by influence
    count_influences = within_eps.sum(axis=1, keepdims=True)
    new_opinions = np.divide(influential.sum(axis=1), count_influences, out=np.copy(op), where=count_influences != 0)
    return new_opinions


def update_opinions_RFM(A, R, op, eps):
    # Compute the absolute differences and reliance thresholds
    diff_matrix = np.abs(op[:, None] - op[None, :])  # |op[i] - op[j]|
    threshold_matrix = R * eps  # R[i, j] * eps

    # Mask where |op[i] - op[j]| <= R[i, j] * eps and connections exist
    mask = (diff_matrix <= threshold_matrix) & (A == 1)

    # Calculate relying individuals and their influence in one step
    relying_values = mask * (op * R)  # R[i, j] * op[j] for valid pairs
    influence_sums = mask * R  # Summed R[i, j] for valid pairs

    # Handle divisions safely: sum values / sum influences
    new_opinions = relying_values.sum(axis=1) / np.maximum(influence_sums.sum(axis=1), 1e-9)
    return new_opinions


def update_reliance(A, R, op, eps, adj):
    differences = np.abs(op[:, None] - op[None, :])
    within_eps = (differences <= R * eps) & A.astype(bool)
    adjustment = adj * (2 * within_eps - 1)  # +adj where true, -adj otherwise
    new_R = np.clip(R + adjustment, 0, 1)  # Ensure bounds
    return new_R

def insert_agent(A, agent_index):
    A[agent_index, :] = 0
    A[agent_index, agent_index] = 1
    return A


def shift_agent_op(curr_opinions, misinfo_agent, manipulate_to, shift_rate):
    agent_idx = misinfo_agent - 1
    if manipulate_to:
        curr_opinions[agent_idx] = min(curr_opinions[agent_idx] + shift_rate, 1)
    else:
        curr_opinions[agent_idx] = max(curr_opinions[agent_idx] - shift_rate, 0)
    return curr_opinions


def linear_shift_after_converge(case, A, curr_R, curr_opinions, epsilon, h, converge_time_phase_1, converge_time_phase_2, manipulate_to, misinfo_agent, shift_rate, round_op_to_decimal):
    if converge_time_phase_1 + 1 >= h:
        new_opinions = update_opinions_RFM(A, curr_R, curr_opinions, epsilon)
        
        if not np.array_equal(np.sort(np.round(new_opinions, round_op_to_decimal)), np.sort(np.round(curr_opinions, round_op_to_decimal))):
            converge_time_phase_1 = h + 1

    else:
        if converge_time_phase_2 == 0:
            converge_time_phase_2 = converge_time_phase_1
            #print(converge_time_phase_1, converge_time_phase_2)
            rounded_curr_op = np.round(curr_opinions, round_op_to_decimal)
            if case == 1:
                misinfo_agent = np.where(rounded_curr_op == (stats.mode(rounded_curr_op).mode))[0][0] + 1
            else:
                mode_indices = np.where(rounded_curr_op == (stats.mode(rounded_curr_op).mode))[0]
                column_A_sums = np.sum(A, axis=0)
                mode_column_A_sums = column_A_sums[mode_indices]
                column_sums = np.sum(curr_R, axis=0)
                mode_column_sums = column_sums[mode_indices]
                if case == 2:
                    misinfo_agent = mode_indices[np.argmax(mode_column_A_sums)] + 1
                elif case == 3:
                    misinfo_agent = mode_indices[np.argmin(mode_column_A_sums)] + 1
                elif case == 4:
                    misinfo_agent = mode_indices[np.argmax(mode_column_sums)] + 1
                elif case == 6:
                    l = len(mode_indices)
                    tmp_A = A[np.ix_(mode_indices, mode_indices)]
                    column_con_sums = np.sum(tmp_A, axis=0)
                    misinfo_agent = mode_indices[np.argmax(column_con_sums)] + 1
                elif case == 7:
                    l = len(mode_indices)
                    tmp_A = A[np.ix_(mode_indices, mode_indices)]
                    column_con_sums = np.sum(tmp_A, axis=0)
                    misinfo_agent = mode_indices[np.argmin(column_con_sums)] + 1
                else:
                    misinfo_agent = mode_indices[np.argmin(mode_column_sums)] + 1
            A = insert_agent(A, misinfo_agent-1)
            print(f'MA: {misinfo_agent}')

        #manipulating to custom value
        curr_opinions = shift_agent_op(curr_opinions, misinfo_agent, manipulate_to, shift_rate)
        
        # IOW-like
        #curr_opinions[misinfo_agent-1] = curr_opinions[misinfo_agent-1] + (1-mu) * (1 - curr_opinions[misinfo_agent-1])

        new_opinions = update_opinions_RFM(A, curr_R, curr_opinions, epsilon)
                
        if not np.array_equal(np.sort(np.round(new_opinions, round_op_to_decimal)), np.sort(np.round(curr_opinions, round_op_to_decimal))):
            converge_time_phase_2 = h + 1
    
    return new_opinions, converge_time_phase_1, converge_time_phase_2, A, misinfo_agent

def choose_agent(curr_opinions, round_op_to_decimal, case, curr_R, A):
    rounded_curr_op = np.round(curr_opinions, round_op_to_decimal)
    if case == 1:
        misinfo_agent = np.where(rounded_curr_op == (stats.mode(rounded_curr_op).mode))[0][0] + 1
    else:
        mode_indices = np.where(rounded_curr_op == (stats.mode(rounded_curr_op).mode))[0]
        column_A_sums = np.sum(A, axis=0)
        mode_column_A_sums = column_A_sums[mode_indices]
        column_sums = np.sum(curr_R, axis=0)
        mode_column_sums = column_sums[mode_indices]
        if case == 2:
            misinfo_agent = mode_indices[np.argmax(mode_column_A_sums)] + 1
        elif case == 3:
            misinfo_agent = mode_indices[np.argmin(mode_column_A_sums)] + 1
        elif case == 4:
            misinfo_agent = mode_indices[np.argmax(mode_column_sums)] + 1
        elif case == 6:
            l = len(mode_indices)
            tmp_A = A[np.ix_(mode_indices, mode_indices)]
            column_con_sums = np.sum(tmp_A, axis=0)
            misinfo_agent = mode_indices[np.argmax(column_con_sums)] + 1
        elif case == 7:
            l = len(mode_indices)
            tmp_A = A[np.ix_(mode_indices, mode_indices)]
            column_con_sums = np.sum(tmp_A, axis=0)
            misinfo_agent = mode_indices[np.argmin(column_con_sums)] + 1
        else:
            misinfo_agent = mode_indices[np.argmin(mode_column_sums)] + 1
    A = insert_agent(A, misinfo_agent-1)
    print(f'MA: {misinfo_agent}')
    return A, misinfo_agent


def linear_shift_before_converge(early_t, shift_rate, case, A, curr_R, curr_opinions, epsilon, h, converge_time_phase_1, converge_time_phase_2, manipulate_to, misinfo_agent, round_op_to_decimal):
    if early_t > h:
        new_opinions = update_opinions_RFM(A, curr_R, curr_opinions, epsilon)
        
        #if not np.array_equal(np.sort(np.round(new_opinions, round_op_to_decimal)), np.sort(np.round(curr_opinions, round_op_to_decimal))):
        if not np.all(np.abs(new_opinions - curr_opinions) < 0.001):
            converge_time_phase_1 = h + 1

    else:
        if converge_time_phase_2 == 0:
            converge_time_phase_2 = converge_time_phase_1
            #print(converge_time_phase_1, converge_time_phase_2)
            #A, misinfo_agent = choose_agent(curr_opinions, round_op_to_decimal, case, curr_R, A)

        #manipulating to custom value
        curr_opinions = shift_agent_op(curr_opinions, misinfo_agent, manipulate_to, shift_rate)

        new_opinions = update_opinions_RFM(A, curr_R, curr_opinions, epsilon)
                
        #if not np.array_equal(np.sort(np.round(new_opinions, round_op_to_decimal)), np.sort(np.round(curr_opinions, round_op_to_decimal))):
        if not np.all(np.abs(new_opinions - curr_opinions) < 0.001):
            converge_time_phase_2 = h + 1
    
    return new_opinions, converge_time_phase_1, converge_time_phase_2, A, misinfo_agent

def read_real_network(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Extract edges, skipping lines starting with '%'
    nodes = int(lines[1].split(' ')[2])
    A = np.zeros((nodes, nodes))
    R = np.zeros((nodes, nodes))
    for line in lines[2:]:
        source, target, weight = map(int, line.strip().split(' '))
        A[source-1][target-1] = 1
        R[source-1][target-1] = weight

    #row_A_sums = np.sum(A, axis=1)
    #for row in range(nodes):
     #   R[row] = R[row]/row_A_sums[row]
    R = R / np.max(R)

    np.fill_diagonal(A, 1)
    np.fill_diagonal(R, 0.5)

    return [A], [R], nodes