from ssl import ALERT_DESCRIPTION_UNRECOGNIZED_NAME
import matplotlib
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
from pgmpy.inference import VariableElimination

alarm_model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Incendiu', 'Alarma'), ('Cutremur', 'Alarma')])

CPD_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
print(CPD_cutremur)
CPD_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99, 0.97],
                                                                        [0.01, 0.03]],
                                                                evidence=['Cutremur'],
                                                                evidence_card=[2])
print(CPD_incendiu)
CPD_alarma = TabularCPD(variable='Alarma', variable_card=2, values=[[0.9999, 0.98, 0.05, 0.02],
                                                                    [0.0001, 0.02, 0.95, 0.98]],
                                                            evidence=['Incendiu', 'Cutremur'],
                                                            evidence_card=[2, 2])
print(CPD_alarma)

alarm_model.add_cpds(CPD_alarma, CPD_cutremur, CPD_incendiu)
print(alarm_model.get_cpds())
print(alarm_model.check_model())

infer = VariableElimination(alarm_model)
result = infer.query(variables=['Cutremur'], evidence={'Alarma': 1})
print(result)

infer = VariableElimination(alarm_model)
result = infer.query(variables=['Incendiu'], evidence={'Alarma': 0})
print(result)