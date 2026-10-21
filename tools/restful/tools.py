from typing import Optional

from bs4 import BeautifulSoup
from rdkit import Chem


def retrival_table_row(html_table):
    """
    Covert the html table to csv table form DrugBank website.
    :param html_table:
    :return:
    """
    soup = BeautifulSoup(html_table, 'html.parser')

    all_rows_data = []

    for row in soup.find_all('tr'):
        tds = row.find_all('td')
        if len(tds) >= 3:
            # 提取药物
            drug_a = tds[0].find('a')
            drug_id = drug_a['href'].split('/')[-1] if drug_a else "N/A"
            drug_name = drug_a.get_text(strip=True) if drug_a else "N/A"

            # 提取类型
            type_div = tds[1].find('div')
            type = type_div.get_text(strip=True) if type_div else "N/A"

            # 提取靶点
            target_a = tds[2].find('a')
            target_id = target_a['href'].split('/')[-1] if target_a else "N/A"
            target_name = target_a.get_text(strip=True) if target_a else "N/A"

            # 组装本行数据
            all_rows_data.append({
                "drug_id": drug_id,
                "drug_name": drug_name,
                "type": type,
                "target_id": target_id,
                "target_name": target_name
            })

    return all_rows_data


def mol2_to_smiles(mol2_file_path: str, isomeric=True) -> Optional[str]:
    mol = Chem.MolFromMol2File(mol2_file_path, sanitize=True, removeHs=True)

    if mol is None:
        return None

    # Convert the molecule object to Canonical SMILES
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)

    return smiles