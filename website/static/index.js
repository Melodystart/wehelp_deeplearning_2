async function prediction() {
  const title = document.getElementById("title").value;
  const response = await fetch("api/model/prediction/" + title, {
    method: "GET",
  });
  const responseJson = await response.json();
  if (response.ok) {
    console.log(responseJson);
  }
}
