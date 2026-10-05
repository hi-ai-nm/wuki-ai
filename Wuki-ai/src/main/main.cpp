#include "llm.cpp"
#include <iostream>

int main() {
    Arch::DRNN model;
    // outRNN 的 neur 是「完整 neur」：15 = 输入层（= allRNN 一次产出的宽度 onceAllOutputNum）、
    // 4 = 隐层、2 = 输出层（= 目标宽度）。hSize 由 set_neur 自己加到首尾，这里不要自己加 h。
    size_t initCode = model.Init({ 15 , 4 , 2 }, { 25 , 15 , 5 }, 15, 15, 6, 3);
    std::cout << "Init: " << initCode << " (" << model.errorBlgig << ")\n";
    if (initCode != 0) {
        return 1;  // 参数不合法就没必要往下跑了
    }

    for (size_t t = 0; t < 10000; t++) {
        auto [loss, code] = model.Train({{ 0.0f , 0.5f }, { 0.5f , 1.0f }}, { 1.5f , 2.0f});
        if (code != 0) {
            std::cout << "Train error: " << code << " (" << model.errorBlgig << ")\n";
            return 2;
        }
        if (t % 100 == 0) {  // 每 100 步看一眼误差，用来确认训练真的在改权重
            std::cout << "step " << t << " loss " << (loss.empty() ? 0.0f : loss[0]) << "\n";
        }
    }

    auto [result, error] = model.Run({{ 0.0f , 0.5f }, { 0.5f , 1.0f }});
    std::cout << "Result (" << result.size() << "): \n";
    for (auto &r : result) {
        std::cout << r << " ";
    }
    std::cout << "\nerror: " << error << " (" << model.errorBlgig << ")\n";

    // ---- 存 / 取往返：存下来 → 读进另一个模型 → 两个模型对同一输入的输出必须逐位相同 ----
    // 放在 exe 旁边（exeDir 取不到就退化成当前目录），这样从哪个工作目录启动都一样
    const std::string binDir = wuki::exeDir();
    const std::string modelPath = (binDir.empty() ? std::string(".") : binDir) + "/drnn.model.json";

    Arch::DRNNSL saveLoad;
    if (!saveLoad.Save(model, modelPath)) {
        std::cout << "Save 失败: " << saveLoad.errorCode << " (" << saveLoad.errorBlgig << ")\n";
        return 3;
    }
    std::cout << "Save ok: " << modelPath << "\n";

    Arch::DRNN loaded;  // 新模型，除了下面这次 Load 之外没碰过任何训练
    if (!saveLoad.Load(loaded, modelPath)) {
        std::cout << "Load 失败: " << saveLoad.errorCode << " (" << saveLoad.errorBlgig << ")\n";
        return 4;
    }
    std::cout << "Load ok\n";

    auto [result2, error2] = loaded.Run({{ 0.0f , 0.5f }, { 0.5f , 1.0f }});
    // 同一份权重、同一套前向 → 应当逐位相同（json 里的 float 是「能精确还原」的写法）
    bool same = (error2 == 0x00 && result2.size() == result.size());
    if (same) {
        for (size_t i = 0; i < result.size(); i++) {
            if (result2[i] != result[i]) same = false;
        }
    }
    std::cout << "往返一致: " << (same ? "yes" : "no") << "\n";
    return same ? 0 : 5;
}
