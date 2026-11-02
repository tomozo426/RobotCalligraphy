#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <QQuickItem>
#include <QQuickWindow>
#include <QTimer>

#include <ignition/gazebo/gui/GuiSystem.hh>
#include <ignition/msgs/image.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/rendering/RenderEngine.hh>
#include <ignition/rendering/ogre2/Ogre2Material.hh>
#include <ignition/rendering/RenderEngineManager.hh>
#include <ignition/rendering/Scene.hh>
#include <ignition/rendering/Visual.hh>
#include <ignition/transport/Node.hh>

#include <OgreHlmsPbsDatablock.h>
#include <OgreStagingTexture.h>
#include <OgreTextureGpu.h>
#include <OgreTextureGpuManager.h>

namespace
{
class TextureUpdateGui final : public ignition::gazebo::GuiSystem
{
  public: TextureUpdateGui()
  {
    if (!this->transportNode.Subscribe(
        "/paper/texture_update", &TextureUpdateGui::OnImage, this))
    {
      std::cerr << "texture GUI: failed to subscribe to /paper/texture_update"
                << std::endl;
    }
    this->renderWindowPoll = new QTimer(this);
    QObject::connect(this->renderWindowPoll, &QTimer::timeout, this,
        [this]() { this->ConnectRenderSignal(); });
    this->renderWindowPoll->start(50);
  }

  private: void ConnectRenderSignal()
  {
    auto *item = this->PluginItem();
    auto *window = item ? item->window() : nullptr;
    if (!window)
      return;
    QObject::connect(window, &QQuickWindow::beforeRendering, this,
        [this]() { this->OnBeforeRendering(); }, Qt::DirectConnection);
    this->renderWindowPoll->stop();
  }

  private: void OnBeforeRendering()
  {
    std::vector<uint8_t> rgb;
    uint32_t step;
    uint64_t sequence;
    {
      std::lock_guard<std::mutex> lock(this->imageMutex);
      if (this->imageSequence == 0 ||
          this->imageSequence == this->uploadedSequence)
      {
        return;
      }
      rgb = this->latestImage;
      step = this->latestStep;
      sequence = this->imageSequence;
    }

    auto *engine = ignition::rendering::RenderEngineManager::Instance()
        ->Engine("ogre2");
    if (!engine)
      return;

    unsigned int uploadedSceneCount = 0;
    uint8_t uploadedMipCount = 0;
    for (unsigned int sceneIndex = 0;
         sceneIndex < engine->SceneCount(); ++sceneIndex)
    {
      auto scene = engine->SceneByIndex(sceneIndex);
      ignition::rendering::VisualPtr visual;
      for (unsigned int nodeIndex = 0; nodeIndex < scene->NodeCount(); ++nodeIndex)
      {
        auto node = scene->NodeByIndex(nodeIndex);
        if (node && node->Name() == "paper::link::surface")
        {
          visual = std::dynamic_pointer_cast<
              ignition::rendering::Visual>(node);
          break;
        }
      }
      if (!visual)
        continue;

      auto material = visual->Material();
      if (!material && visual->GeometryCount() > 0)
        material = visual->GeometryByIndex(0)->Material();
      auto *ogreMaterial = material
          ? dynamic_cast<ignition::rendering::Ogre2Material *>(material.get())
          : nullptr;
      auto *datablock = ogreMaterial ? ogreMaterial->Datablock() : nullptr;
      auto *texture = datablock
          ? datablock->getTexture(Ogre::PBSM_DIFFUSE) : nullptr;
      if (!texture || texture->getWidth() != 256 ||
          texture->getHeight() != 256)
      {
        continue;
      }

      const auto format = texture->getPixelFormat();
      if (format != Ogre::PFG_RGBA8_UNORM &&
          format != Ogre::PFG_RGBA8_UNORM_SRGB)
      {
        continue;
      }
      std::vector<uint8_t> rgba(256 * 256 * 4);
      for (uint32_t y = 0; y < 256; ++y)
      {
        const auto *source = rgb.data() + y * step;
        auto *destination = rgba.data() + y * 256 * 4;
        for (uint32_t x = 0; x < 256; ++x)
        {
          destination[x * 4 + 0] = source[x * 3 + 0];
          destination[x * 4 + 1] = source[x * 3 + 1];
          destination[x * 4 + 2] = source[x * 3 + 2];
          destination[x * 4 + 3] = 255;
        }
      }

      auto *textureManager = texture->getTextureManager();
      if (!this->stagingTexture)
      {
        this->stagingTexture = textureManager->getStagingTexture(
            256, 256, 1, 1, format);
      }
      if (!this->stagingTexture)
        continue;

      uint32_t mipWidth = 256;
      uint32_t mipHeight = 256;
      const auto mipCount = texture->getNumMipmaps();
      bool uploadFailed = false;
      for (uint8_t mip = 0; mip < mipCount; ++mip)
      {
        this->stagingTexture->startMapRegion();
        auto box = this->stagingTexture->mapRegion(
            mipWidth, mipHeight, 1, 1, format);
        if (!box.data)
        {
          this->stagingTexture->stopMapRegion();
          uploadFailed = true;
          break;
        }
        for (uint32_t y = 0; y < mipHeight; ++y)
        {
          auto *destination = static_cast<uint8_t *>(box.data) +
              y * box.bytesPerRow;
          std::memcpy(destination,
              rgba.data() + y * mipWidth * 4, mipWidth * 4);
        }
        this->stagingTexture->stopMapRegion();
        Ogre::TextureBox destinationBox(
            mipWidth, mipHeight, 1, 1, 4, mipWidth * 4,
            mipWidth * mipHeight * 4);
        destinationBox.sliceStart = texture->getInternalSliceStart();
        this->stagingTexture->upload(box, texture, mip, nullptr,
                                     &destinationBox);

        if (mip + 1 < mipCount)
        {
          const uint32_t nextWidth = std::max(1u, mipWidth / 2);
          const uint32_t nextHeight = std::max(1u, mipHeight / 2);
          std::vector<uint8_t> nextRgba(nextWidth * nextHeight * 4);
          for (uint32_t y = 0; y < nextHeight; ++y)
          {
            for (uint32_t x = 0; x < nextWidth; ++x)
            {
              for (uint32_t channel = 0; channel < 4; ++channel)
              {
                uint32_t sum = 0;
                for (uint32_t dy = 0; dy < 2; ++dy)
                {
                  for (uint32_t dx = 0; dx < 2; ++dx)
                  {
                    const auto sourceIndex =
                        ((y * 2 + dy) * mipWidth + x * 2 + dx) * 4 + channel;
                    sum += rgba[sourceIndex];
                  }
                }
                nextRgba[(y * nextWidth + x) * 4 + channel] =
                    static_cast<uint8_t>(sum / 4);
              }
            }
          }
          rgba = std::move(nextRgba);
          mipWidth = nextWidth;
          mipHeight = nextHeight;
        }
      }
      if (uploadFailed)
        continue;
      ++uploadedSceneCount;
      uploadedMipCount = mipCount;
    }

    if (uploadedSceneCount == 0)
      return;
    this->uploadedSequence = sequence;
    if (sequence == 1)
      std::cerr << "texture GUI: uploaded image sequence " << sequence
                << " to " << uploadedSceneCount << " GUI scene(s), "
                << static_cast<int>(uploadedMipCount) << " mip levels"
                << std::endl;
  }

  private: void OnImage(const ignition::msgs::Image &_image)
  {
    if (_image.width() != 256 || _image.height() != 256 ||
        _image.pixel_format_type() != ignition::msgs::RGB_INT8 ||
        _image.step() < _image.width() * 3 ||
        _image.data().size() < _image.step() * _image.height())
    {
      return;
    }

    std::lock_guard<std::mutex> lock(this->imageMutex);
    this->latestImage.assign(_image.data().begin(), _image.data().end());
    this->latestStep = _image.step();
    ++this->imageSequence;
  }

  private: ignition::transport::Node transportNode;
  private: std::mutex imageMutex;
  private: std::vector<uint8_t> latestImage;
  private: uint32_t latestStep{0};
  private: uint64_t imageSequence{0};
  private: uint64_t uploadedSequence{0};
  private: QTimer *renderWindowPoll{nullptr};
  private: Ogre::StagingTexture *stagingTexture{nullptr};
};
}

IGNITION_ADD_PLUGIN(TextureUpdateGui,
    ignition::gui::Plugin)
IGNITION_ADD_PLUGIN_ALIAS(TextureUpdateGui,
    "texture_probe::TextureUpdateGui")