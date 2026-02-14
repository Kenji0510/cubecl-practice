use anyhow::Result;
use pcd_rs::reader::Reader;
use pcd_rs::{PcdDeserialize, PcdSerialize};

#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub fn load_pcd_xyz(file_path: &str) -> Result<Vec<PointXYZ>> {
    let reader = match Reader::open(file_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open PCD file: {}", e);
            return Err(anyhow::anyhow!("Failed to open PCD file: {}", e));
        }
    };

    let points: Vec<PointXYZ> = match reader.collect() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to read PCD data: {}", e);
            return Err(anyhow::anyhow!("Failed to read PCD data: {}", e));
        }
    };

    Ok(points)
}

pub fn save_xyz_pcd(points: &[PointXYZ], file_path: &str) -> Result<()> {
    let mut writer = pcd_rs::WriterInit {
        width: 1,
        height: points.len() as u64,
        viewpoint: Default::default(),
        data_kind: pcd_rs::DataKind::Ascii,
        schema: None,
    }
    .create(file_path)?;

    for point in points {
        writer.push(point)?;
    }
    writer.finish()?;

    Ok(())
}
