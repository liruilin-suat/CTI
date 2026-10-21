import requests
import time


def get_protein_name(ensp_id):

    response = requests.post(
        "https://rest.uniprot.org/idmapping/run",
        data={
            "from": "Ensembl_Protein",
            "to": "UniProtKB",
            "ids": ensp_id
        }
    )
    response.raise_for_status()

    job_id = response.json()["jobId"]

    while True:
        response = requests.get(
            f"https://rest.uniprot.org/idmapping/status/{job_id}"
        )
        response.raise_for_status()
        data = response.json()

        if "jobStatus" not in data:
            break

        time.sleep(1)

    # 获取 Mapping 结果
    response = requests.get(
        f"https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}"
    )
    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        return None

    uniprot_id = results[0]["to"]["primaryAccession"]

    # 获取 Protein Name
    response = requests.get(
        f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    )
    response.raise_for_status()

    protein_data = response.json()

    try:
        protein_name = (
            protein_data["proteinDescription"]
            ["recommendedName"]
            ["fullName"]
            ["value"]
        )
    except KeyError:
        protein_name = None

    return {
        "ENSP_ID": ensp_id,
        "UniProt_ID": uniprot_id,
        "Protein_Name": protein_name
    }


result = get_protein_name("ENSP00000263100")
print(result)