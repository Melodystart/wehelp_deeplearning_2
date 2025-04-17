async function prediction() {
  const title = document.getElementById("title").value;
  if (title.length == 0) {
    alert("請輸入文章標題");
    return;
  }
  document.body.style.cursor = "wait";
  document.querySelector("#submit").style.cursor = "not-allowed";
  document.querySelector(".result-container").style.display = "none";
  document.querySelector(".loading").style.display = "flex";

  const response = await fetch(
    "/api/model/prediction/?title=" + encodeURIComponent(title),
    {
      method: "GET",
    }
  );

  const result = await response.json();
  if (response.ok) {
    document.body.style.cursor = "default";
    document.querySelector("#submit").style.cursor = "pointer";
    console.log(result);
    let maxKey = "";
    let maxValue = -0.01;

    for (const key in result) {
      if (result[key] > maxValue) {
        maxValue = result[key];
        maxKey = key;
      }
    }

    const boards = Object.keys(result);
    const labels = document.querySelector(".labels");
    labels.innerHTML = "";
    boards.forEach((item) => {
      const label = document.createElement("button");
      label.id = item;
      label.classList.add("label");
      label.onclick = () => feedback(item);
      label.textContent = item;
      labels.appendChild(label);
    });

    document.querySelector(".loading").style.display = "none";
    document.querySelector(".feeback-container").style.display = "none";
    document.querySelector(".result-container").style.display = "flex";
    document.querySelector(".result").textContent =
      maxKey + "    " + (maxValue * 100).toFixed(2) + "%";
    document.querySelector(".labels").style.display = "flex";
    document.querySelector(".choose").style.display = "flex";
  } else {
    document.body.style.cursor = "default";
    document.querySelector("#submit").style.cursor = "pointer";
    alert("預測發生錯誤");
  }
}

async function feedback(label) {
  const title = document.getElementById("title").value;

  const response = await fetch("/api/model/feedback/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title, label }),
  });

  const result = await response.json();
  if (result.ok) {
    document.querySelector(".labels").style.display = "none";
    document.querySelector(".choose").style.display = "none";
    document.querySelector(".feeback-container").style.display = "flex";
    document.querySelector(".feeback").textContent = label;
  } else {
    alert("紀錄回饋失敗");
  }
}
