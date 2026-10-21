import os
from typing import Optional, Union
import requests
import re
import json
from requests.adapters import HTTPAdapter
from urllib3 import Retry


def create_retry_session(
        retries=3,
        backoff_factor=1,
        status_forcelist=(500, 502, 503, 504, 429)
):
    session = requests.Session()

    # 配置重试策略
    retry_strategy = Retry(
        total=retries,  # 总重试次数
        backoff_factor=backoff_factor,  # 重试延迟间隔系数 (1s, 2s, 4s...)
        status_forcelist=status_forcelist,  # 针对哪些 HTTP 状态码触发重试
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


############################
# restful api for TCM-suite #
############################
def get_ensembl_sequences(ensembl_ids: list, retries: int = 5) -> dict:
    """
    Batch retrieve FASTA sequences using the Ensembl REST API.
    :param ensembl_ids: A list of Ensembl IDs (e.g., ['ENSP00000139618', 'ENSP00000141510'])
    :param retries: number of retransmissions
    :return: A dictionary where keys are Ensembl IDs and values are their corresponding sequences.

    Example:
        >> res = get_ensembl_sequences(['ENSP00000139618', 'ENSP00000141510',...])
        >> print(res)
        {"ENSP00000139618": "MPVLS...", "ENSP00000141510": "MPVLS..."}
    """
    url = "https://rest.ensembl.org/sequence/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Session
    session = create_retry_session(retries)
    payload = {"ids": ensembl_ids, "type": "protein"}

    seq_map = {}

    try:
        response = session.post(
            url, headers=headers, data=json.dumps(payload), timeout=30
        )
        if not response.ok:
            return seq_map

        results = response.json()

        if isinstance(results, list):
            for item in results:
                if item and "seq" in item:
                    seq_map[item["id"]] = item["seq"]
        elif isinstance(results, dict) and "error" in results:
            print(f"⚠️ [API 警告] 服务器返回错误信息: {results['error']}")

    except requests.exceptions.RequestException as e:
        print(f"💥 [严重网络错误] 无法连接到 Ensembl 服务器: {e}")

    finally:
        # 关闭会话释放资源
        session.close()
    return seq_map


def request_ensembl_protein_page(ensembl_id: str, retries: int = 5) -> str:
    """

    :param ensembl_id:
    :param retries: number of retransmissions
    :return:
    """
    url = f"https://www.ensembl.org/Homo_sapiens/Transcript/Idhistory/Protein?t={ensembl_id}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0"
    }

    # Session
    session = create_retry_session(retries)

    content = ""
    try:
        response = session.get(
            url, headers=headers, timeout=30
        )
        if response.status_code == 200:
            content = response.text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return content


############################
# restful api for  Symmap  #
############################
def get_seq_by_uniprot_id(id_, retries: int = 5) -> str:
    """
    Query protein sequence from uniprot.
    :param id_: 'string' uniprot id of protein
    :param retries: number of retransmissions
    :return 'string' protein sequence
    """
    url = "https://www.uniprot.org/uniprot/%s.fasta" % id_
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Session
    session = create_retry_session(retries)

    seq = ""
    try:
        data = session.get(url, headers=headers, timeout=30)
        if data.status_code == 200:
            seq = "".join(data.text.split("\n")[1:])
    except requests.exceptions.RequestException as e:
        print("Getting protein sequence occur error: {}".format(e))
    finally:
        # 关闭会话释放资源
        session.close()
    return seq


def get_smiles_by_cid(cid, isomeric: bool = True, retries: int = 5) -> str:
    """
    Retrieve the Canonical SMILES using the PubChem CID.
    :param cid: 'string' Canonical CID
    :param isomeric: number of retransmissions
    :param retries:
    :return: 'string' smiles or ""
    """
    if isomeric:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/smiles/JSON"
    else:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Session
    session = create_retry_session(retries)

    smiles = ""
    try:
        response = session.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # retrieve SMILES
            if isomeric:
                smiles = data['PropertyTable']['Properties'][0]['SMILES']
            else:
                smiles = data['PropertyTable']['Properties'][0]['ConnectivitySMILES']
            return smiles

    except requests.exceptions.RequestException as e:
        print(f"Error occurred while looking up CID {cid}: {e}")

    finally:
        # 关闭会话释放资源
        session.close()

    return smiles


def get_smiles_by_cas(cas_number, isomeric: bool = True, retries: int = 5) -> str:
    """
    Retrieves the SMILES string for a given CAS number using PubChem.

    :param
        cas_number (str): The CAS number of the compound.
        isomeric: isomeric SMILES
        retries: number of retransmissions

    :returns
        str: The corresponding SMILES string if found, otherwise an error message.
    """
    # PubChem 接口：通过 synonym (CAS) 获取 CID，再直接拉取 CanonicalSMILES
    if isomeric:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/property/smiles/JSON"
    else:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/property/CanonicalSMILES/JSON"

    headers = {
        # 加上 User-Agent 避免被 NCBI 服务器当作恶意爬虫拦截
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/Requests"
    }
    # Session
    session = create_retry_session(retries)

    smiles = ""
    try:
        response = session.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # retrieve SMILES
            if isomeric:
                smiles = data['PropertyTable']['Properties'][0]['SMILES']
            else:
                smiles = data['PropertyTable']['Properties'][0]['ConnectivitySMILES']
            return smiles

    except requests.exceptions.RequestException as e:
        print(f"Error occurred while looking up CAS {cas_number}: {e}")

    finally:
        # 关闭会话释放资源
        session.close()

    return smiles

def get_cid_by_inchikey(inchikey: str) -> Optional[str]:
    """
    Retrieves the cid by inchikey from PubChem database.
    :param inchikey: string
    :return: cid or None

    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/rdf/inchikey/{inchikey}.html"

    html_content = requests.get(url).text

    pattern = r'pubchem/compound/CID(\d+)'

    # find all matches using re.findall
    cids = re.findall(pattern, html_content)

    if cids:
        return cids[0]
    else:
        return None

def get_mol(mol_ids: list, save_dir: str, retries: int = 5) -> list:
    """
    Retrieve Mol files from the TCMSP website.
    :param mol_ids: A list of TCMSP molecule IDs (e.g., ['MOL000001', 'MOL000002'])
    :param save_dir: Path to the directory where the downloaded Mol files will be saved.
    :param retries: number of retransmissions
    :return: None (or a dictionary of saved file paths, depending on your implementation)
    """

    # create session
    session = create_retry_session(retries)

    # request header
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 Chrome/125.0 Safari/537.36"
        )
    }
    # record successful
    success = 0
    failed = []

    for _, mol_id in mol_ids:
        # save file
        save_path = os.path.join(save_dir, f"{mol_id}.mol2")

        if os.path.exists(save_path):
            continue

        url = f"https://www.tcmsp-e.com/tcmspmol/{mol_id}.mol2"

        # represents whether getting the mol2 file was successful
        ok = False

        try:

            response = session.get(
                url,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:

                if response.content.startswith(b"@<TRIPOS>"):

                    with open(save_path, "wb") as f:
                        f.write(response.content)

                    success += 1
                    print(f"[{success}] success: {mol_id}")

                    ok = True
                    break

                else:
                    print(f"{mol_id}: Returned content is not a valid mol2 file")

            else:
                print(f"{mol_id}: HTTP {response.status_code}")

        except Exception as e:
            print(f"{mol_id}: failed -> {e}")

        if not ok:
            failed.append(mol_id)

    return failed